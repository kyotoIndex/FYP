"""Exclusive host-time accounting for a cost replay.

The caller declares each operation label's category and must surround the
entire interval it wants to account for. Nested spans charge elapsed host time
only to the innermost active label. No GPU package is imported; a caller may
inject a device synchronization callback at timing boundaries.

This module does not decide which concrete runtime operations are necessary.
It does not add device-event durations to host time or alter random state.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import time
from types import MappingProxyType
from typing import Callable, Iterator, Mapping


NECESSARY_ALGORITHM = "necessary_algorithm"
RESEARCH_OBSERVATION = "research_observation"
POST_BUDGET_DIAGNOSTIC = "post_budget_diagnostic"
UNCLASSIFIED_GAP = "unclassified_gap"
MEASUREMENT_OVERHEAD = "measurement_overhead"
ENVIRONMENT_WAIT = "environment_wait"
CATEGORIES = frozenset({NECESSARY_ALGORITHM, RESEARCH_OBSERVATION,
                        POST_BUDGET_DIAGNOSTIC, UNCLASSIFIED_GAP,
                        MEASUREMENT_OVERHEAD, ENVIRONMENT_WAIT})
DEPLOYED_PREFIX = "deployed_prefix"
DIAGNOSTIC_AFTER_BUDGET = "diagnostic_after_budget"
BUCKETS = frozenset({DEPLOYED_PREFIX, DIAGNOSTIC_AFTER_BUDGET})
SCHEMA = "pcla-exclusive-host-timing-2"

# Each leaf has exactly one of the seven required-time groups, or an explicit
# exclusion.  A nested leaf replaces its parent's charge; it is never added
# to both groups.  Cohort-internal mixed operations require separate hooks.
REQUIRED_GROUPS = (
    "task_initialization", "state_and_optimizer_setup", "client_training",
    "scoring_and_shared_information", "coordinate_and_server_decision",
    "message_processing", "budget_and_return_control",
)
COST_LABEL_GROUPS = {
    **{group: group for group in REQUIRED_GROUPS},
    "local_training": "client_training",
    "local_setup": "state_and_optimizer_setup",
}
COST_LABEL_CATEGORIES = {
    **{label: NECESSARY_ALGORITHM for label in COST_LABEL_GROUPS},
    "envelope": UNCLASSIFIED_GAP,
    "round_unclassified": UNCLASSIFIED_GAP,
    "prefix_capture": RESEARCH_OBSERVATION,
    "recipe_dev_evaluation": RESEARCH_OBSERVATION,
    "ema_update": RESEARCH_OBSERVATION,
    "ema_evaluation": RESEARCH_OBSERVATION,
    "half_budget_clone": RESEARCH_OBSERVATION,
    "technical_logits_check": RESEARCH_OBSERVATION,
    "technical_equivalence_check": RESEARCH_OBSERVATION,
    "checkpoint_and_audit": RESEARCH_OBSERVATION,
    "provenance_and_hashing": RESEARCH_OBSERVATION,
    "post_budget_diagnostic": POST_BUDGET_DIAGNOSTIC,
    "measurement_overhead": MEASUREMENT_OVERHEAD,
    "environment_wait": ENVIRONMENT_WAIT,
}


def required_breakdown(record: Mapping) -> dict[str, float]:
    """Project disjoint prefix leaves onto the seven protocol groups."""
    if record.get("partial"):
        raise ValueError("partial failure timing has no accepted required-time breakdown")
    labels = record["bucket_label_seconds"][DEPLOYED_PREFIX]
    if set(labels) != set(COST_LABEL_CATEGORIES):
        raise ValueError("cost timing labels differ from declared schema")
    return {group: sum(labels[label] for label, assigned in COST_LABEL_GROUPS.items()
                       if assigned == group) for group in REQUIRED_GROUPS}


class UnclosedIntervalError(RuntimeError):
    """The ledger cannot be persisted while a timing interval is active."""


class ExecutionTimingLedger:
    """Accumulate disjoint host seconds under explicitly classified labels.

    ``clock`` defaults to ``time.perf_counter``. ``synchronize``, if supplied,
    runs before every clock sample, so its wait is included in the currently
    active label at a closing boundary. Neither callback is invoked by a
    rejected unknown label or a rejected out-of-order stop.

    The ledger is intended for one execution thread. Persist only after all
    spans close; a resumed ledger starts a fresh wall interval and retains the
    accumulated totals.
    """

    def __init__(
        self,
        label_categories: Mapping[str, str],
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        labels = dict(label_categories)
        if not labels or any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("declare at least one nonempty operation label")
        unknown = {category for category in labels.values()
                   if not isinstance(category, str) or category not in CATEGORIES}
        if unknown:
            raise ValueError(f"unknown timing categories: {sorted(unknown)}")
        if ("measurement_overhead" in labels and
                labels["measurement_overhead"] != MEASUREMENT_OVERHEAD):
            raise ValueError("measurement_overhead label has a reserved category")
        if not callable(clock) or (synchronize is not None and not callable(synchronize)):
            raise TypeError("clock and synchronize must be callable")
        self.label_categories = MappingProxyType(labels)
        self._clock = clock
        self._synchronize = synchronize
        self._seconds = {bucket: {label: 0.0 for label in labels} for bucket in BUCKETS}
        self._stack: list[tuple[int, str, str, int | None]] = []
        self._events: list[dict] = []
        self._completed_spans = {label: 0 for label in labels}
        self._envelope_token: int | None = None
        self._envelope_bounds: list[float | None] = [None, None]
        self._last_sample: float | None = None
        self._next_token = 1

    @property
    def active_labels(self) -> tuple[str, ...]:
        return tuple(label for _, label, _, _ in self._stack)

    def _sample(self) -> float:
        if self._synchronize is not None:
            self._synchronize()
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("clock returned a non-finite value")
        return now

    def _advance(self, now: float) -> None:
        if self._last_sample is not None:
            elapsed = now - self._last_sample
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("clock elapsed time is invalid")
            if self._stack:
                _, label, bucket, round_id = self._stack[-1]
                self._seconds[bucket][label] += elapsed
                if elapsed:
                    self._events.append({"label": label, "category": self.label_categories[label],
                                         "bucket": bucket, "round": round_id,
                                         "start": self._last_sample, "end": now,
                                         "seconds": elapsed})
        self._last_sample = now

    def _charge_bookkeeping(self, boundary: float) -> None:
        """Separate Python ledger bookkeeping from the next algorithm leaf."""
        if "measurement_overhead" not in self.label_categories or not self._stack:
            return
        end = float(self._clock())  # No second device sync: this is host bookkeeping.
        if not math.isfinite(end) or end < boundary:
            raise ValueError("clock elapsed time is invalid")
        duration = end - boundary
        _, _, bucket, round_id = self._stack[-1]
        self._seconds[bucket]["measurement_overhead"] += duration
        if duration:
            self._events.append({"label": "measurement_overhead",
                                 "category": MEASUREMENT_OVERHEAD,
                                 "bucket": bucket, "round": round_id,
                                 "start": boundary, "end": end,
                                 "seconds": duration})
        self._last_sample = end

    def start(self, label: str, *, bucket: str | None = None,
              round_id: int | None = None) -> int:
        """Open a label; nested spans inherit their parent's budget bucket."""
        if label not in self.label_categories:
            raise ValueError(f"unknown timing label: {label!r}")
        if label == "envelope" and (self._stack or self._envelope_bounds[0] is not None):
            raise ValueError("cost envelope must be unique and outermost")
        if bucket is None:
            bucket = self._stack[-1][2] if self._stack else DEPLOYED_PREFIX
        if bucket not in BUCKETS:
            raise ValueError(f"unknown timing bucket: {bucket!r}")
        if round_id is None and self._stack:
            round_id = self._stack[-1][3]
        if round_id is not None and (type(round_id) is not int or round_id < 0):
            raise ValueError("round_id must be a nonnegative integer")
        now = self._sample()
        self._advance(now)
        token = self._next_token
        self._next_token += 1
        self._stack.append((token, label, bucket, round_id))
        if label == "envelope":
            self._envelope_bounds[0] = now
        self._charge_bookkeeping(now)
        return token

    def stop(self, token: int) -> None:
        """Close the innermost label, charging time through this boundary."""
        if not self._stack or self._stack[-1][0] != token:
            raise RuntimeError("timing intervals must close in LIFO order")
        now = self._sample()
        self._advance(now)
        _, label, _, _ = self._stack.pop()
        self._completed_spans[label] += 1
        if label == "envelope":
            self._envelope_bounds[1] = now
        if not self._stack:
            self._last_sample = None
        else:
            self._charge_bookkeeping(now)

    def begin_envelope(self) -> None:
        if self._envelope_token is not None or self._completed_spans.get("envelope", 0):
            raise RuntimeError("cost envelope may begin only once")
        self._envelope_token = self.start("envelope", bucket=DEPLOYED_PREFIX)

    def end_envelope(self) -> None:
        if self._envelope_token is None:
            raise RuntimeError("cost envelope has not begun")
        self.stop(self._envelope_token)
        self._envelope_token = None

    @contextmanager
    def span(self, label: str, *, bucket: str | None = None,
             round_id: int | None = None) -> Iterator[None]:
        """Close this interval even if the measured operation raises."""
        token = self.start(label, bucket=bucket, round_id=round_id)
        try:
            yield
        finally:
            self.stop(token)

    def snapshot(self) -> dict:
        """Serialize settled totals; reject partially measured intervals."""
        if self._stack:
            raise UnclosedIntervalError(f"open timing labels: {self.active_labels}")
        return self._record()

    def partial_snapshot(self) -> dict:
        """Preserve measured failure events without inventing a closed prefix."""
        sample_error = None
        if self._stack:
            try:
                self._advance(self._sample())
            except BaseException as exc:
                sample_error = f"{type(exc).__name__}: {exc}"
        record = self._record()
        record.update({"partial": True,
                       "open_intervals": [
                           {"label": label, "bucket": bucket, "round": round_id}
                           for _, label, bucket, round_id in self._stack],
                       "tail_unmeasured": sample_error is not None,
                       "capture_sample_error": sample_error})
        return record

    def _record(self) -> dict:
        bucket_label_seconds = {bucket: dict(self._seconds[bucket]) for bucket in sorted(BUCKETS)}
        label_seconds = {label: sum(self._seconds[bucket][label] for bucket in BUCKETS)
                         for label in self.label_categories}
        category_seconds = {category: sum(value for label, value in label_seconds.items()
                                          if self.label_categories[label] == category)
                            for category in sorted(CATEGORIES)}
        bucket_category_seconds = {
            bucket: {category: sum(value for label, value in bucket_label_seconds[bucket].items()
                                   if self.label_categories[label] == category)
                     for category in sorted(CATEGORIES)}
            for bucket in sorted(BUCKETS)
        }
        return {
            "schema": SCHEMA,
            "label_categories": dict(self.label_categories),
            "bucket_label_seconds": bucket_label_seconds,
            "bucket_category_seconds": bucket_category_seconds,
            "label_seconds": label_seconds,
            "category_seconds": category_seconds,
            "total_seconds": sum(label_seconds.values()),
            "events": list(self._events),
            "completed_spans": dict(self._completed_spans),
            "envelope_bounds": list(self._envelope_bounds),
        }

    def coverage(self, envelope_label: str) -> dict:
        """Report the observed unclassified gap against Pro's per-run limit.

        The caller must open one outer envelope from warm W0 through the return
        decision.  This method reports evidence; it never fills missing time.
        """
        record = self.snapshot()
        if self.label_categories.get(envelope_label) != UNCLASSIFIED_GAP:
            raise ValueError("envelope label must be classified as unclassified_gap")
        if self._completed_spans[envelope_label] != 1:
            raise ValueError("exactly one completed warm-W0-to-return envelope is required")
        lower, upper = self._envelope_bounds
        if lower is None or upper is None:
            raise ValueError("cost envelope boundary is missing")
        cursor = lower
        for event in record["events"]:
            if event["bucket"] != DEPLOYED_PREFIX:
                continue
            if (event["start"] < lower or event["end"] > upper or
                    not math.isclose(event["start"], cursor, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError("deployed events overlap or leave an envelope hole")
            cursor = event["end"]
        if not math.isclose(cursor, upper, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("deployed events do not reach the envelope end")
        gap = record["bucket_category_seconds"][DEPLOYED_PREFIX][UNCLASSIFIED_GAP]
        elapsed = record["bucket_category_seconds"][DEPLOYED_PREFIX]
        if elapsed[POST_BUDGET_DIAGNOSTIC]:
            raise ValueError("post-budget diagnostic entered the deployed prefix")
        total = sum(elapsed.values())
        if not math.isclose(total, upper - lower, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("timing events do not cover the cost envelope")
        threshold = max(0.05, 0.001 * total)
        return {"envelope_seconds": total, "unclassified_seconds": gap,
                "threshold_seconds": threshold, "passes": gap <= threshold}

    @classmethod
    def from_snapshot(
        cls,
        record: Mapping,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronize: Callable[[], None] | None = None,
    ) -> ExecutionTimingLedger:
        """Restore accumulated seconds without charging the downtime gap."""
        if record.get("schema") != SCHEMA:
            raise ValueError("unsupported timing snapshot schema")
        if record.get("partial"):
            raise ValueError("partial failure timing cannot be restored as a settled ledger")
        ledger = cls(record["label_categories"], clock=clock,
                     synchronize=synchronize)
        seconds = record["label_seconds"]
        if set(seconds) != set(ledger.label_categories):
            raise ValueError("snapshot labels do not match declared labels")
        for label, value in seconds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid seconds for {label!r}")
        bucket_seconds = record["bucket_label_seconds"]
        if set(bucket_seconds) != BUCKETS or any(
            set(bucket_seconds[bucket]) != set(ledger.label_categories) for bucket in BUCKETS
        ):
            raise ValueError("snapshot bucket labels do not match declared labels")
        for bucket in BUCKETS:
            for label, value in bucket_seconds[bucket].items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid seconds for {bucket}/{label}")
        ledger._seconds = {
            bucket: {label: float(value) for label, value in bucket_seconds[bucket].items()}
            for bucket in BUCKETS
        }
        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError("snapshot event table is missing")
        event_totals = {bucket: {label: 0.0 for label in ledger.label_categories}
                        for bucket in BUCKETS}
        for event in events:
            if not isinstance(event, dict) or event.get("label") not in ledger.label_categories:
                raise ValueError("invalid timing event label")
            label, bucket = event["label"], event.get("bucket")
            if bucket not in BUCKETS or event.get("category") != ledger.label_categories[label]:
                raise ValueError("invalid timing event category or bucket")
            start, end, duration = (event.get(key) for key in ("start", "end", "seconds"))
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in (start, end, duration)) or end < start or duration < 0 or not math.isclose(
                       end - start, duration, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("invalid timing event interval")
            round_id = event.get("round")
            if round_id is not None and (type(round_id) is not int or round_id < 0):
                raise ValueError("invalid timing event round")
            event_totals[bucket][label] += duration
        if any(not math.isclose(event_totals[bucket][label], ledger._seconds[bucket][label],
                                rel_tol=1e-12, abs_tol=1e-9)
               for bucket in BUCKETS for label in ledger.label_categories):
            raise ValueError("snapshot events do not conserve bucket label seconds")
        ledger._events = list(events)
        counts = record.get("completed_spans")
        if (not isinstance(counts, dict) or set(counts) != set(ledger.label_categories)
                or any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("invalid completed span counts")
        ledger._completed_spans = dict(counts)
        bounds = record.get("envelope_bounds")
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(x is not None and (isinstance(x, bool) or
                    not isinstance(x, (int, float)) or not math.isfinite(x)) for x in bounds) or
                (bounds[0] is None) != (bounds[1] is None) or
                (bounds[0] is not None and bounds[1] < bounds[0])):
            raise ValueError("invalid envelope bounds")
        ledger._envelope_bounds = list(bounds)
        expected = ledger.snapshot()
        for label in ledger.label_categories:
            if not math.isclose(seconds[label], expected["label_seconds"][label],
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("snapshot label totals do not conserve bucket seconds")
        for field in ("total_seconds",):
            if not math.isclose(record[field], expected[field], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("snapshot total does not conserve label seconds")
        categories = record["category_seconds"]
        if set(categories) != CATEGORIES or any(
            not math.isclose(categories[category], expected["category_seconds"][category],
                             rel_tol=1e-12, abs_tol=1e-9)
            for category in CATEGORIES
        ):
            raise ValueError("snapshot category totals do not conserve label seconds")
        bucket_categories = record["bucket_category_seconds"]
        if set(bucket_categories) != BUCKETS or any(
            set(bucket_categories[bucket]) != CATEGORIES or any(
                not math.isclose(bucket_categories[bucket][category],
                                 expected["bucket_category_seconds"][bucket][category],
                                 rel_tol=1e-12, abs_tol=1e-9)
                for category in CATEGORIES
            )
            for bucket in BUCKETS
        ):
            raise ValueError("snapshot bucket categories do not conserve label seconds")
        return ledger
