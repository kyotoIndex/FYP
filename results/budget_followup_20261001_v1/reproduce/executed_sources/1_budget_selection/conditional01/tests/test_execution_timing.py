import copy
import json
import unittest

from pcla_research_v7.execution_timing import (
    DEPLOYED_PREFIX,
    DIAGNOSTIC_AFTER_BUDGET,
    COST_LABEL_CATEGORIES,
    ExecutionTimingLedger,
    MEASUREMENT_OVERHEAD,
    NECESSARY_ALGORITHM,
    POST_BUDGET_DIAGNOSTIC,
    RESEARCH_OBSERVATION,
    UNCLASSIFIED_GAP,
    UnclosedIntervalError,
    required_breakdown,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


LABELS = {
    "run": NECESSARY_ALGORITHM,
    "wire": NECESSARY_ALGORITHM,
    "evaluation": RESEARCH_OBSERVATION,
    "diagnostic": POST_BUDGET_DIAGNOSTIC,
}


class ExecutionTimingTests(unittest.TestCase):
    def test_bookkeeping_is_explicit_and_does_not_enter_required_time(self):
        class TickClock:
            now = 0.0

            def __call__(self):
                current = self.now
                self.now += 0.1
                return current

        ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES, clock=TickClock())
        ledger.begin_envelope()
        with ledger.span("client_training", round_id=1):
            pass
        ledger.end_envelope()
        record = ledger.snapshot()
        self.assertGreater(record["category_seconds"][MEASUREMENT_OVERHEAD], 0)
        self.assertEqual(record["label_seconds"]["local_setup"], 0)
        self.assertAlmostEqual(record["total_seconds"], ledger.coverage("envelope")["envelope_seconds"])

    def test_partial_failure_keeps_open_interval_and_rejects_restore(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES, clock=clock)
        ledger.begin_envelope()
        ledger.start("client_training", round_id=4)
        clock.advance(3)
        partial = ledger.partial_snapshot()
        self.assertTrue(partial["partial"])
        self.assertFalse(partial["tail_unmeasured"])
        self.assertEqual([x["label"] for x in partial["open_intervals"]],
                         ["envelope", "client_training"])
        self.assertEqual(partial["label_seconds"]["client_training"], 3)
        self.assertEqual(partial["events"][-1]["round"], 4)
        with self.assertRaisesRegex(ValueError, "partial failure"):
            ExecutionTimingLedger.from_snapshot(partial, clock=clock)
        with self.assertRaisesRegex(ValueError, "partial failure"):
            required_breakdown(partial)

    def test_partial_failure_marks_unsampled_tail_if_device_sync_fails(self):
        clock = FakeClock()
        sync_calls = 0

        def sync():
            nonlocal sync_calls
            sync_calls += 1
            if sync_calls == 3:
                raise RuntimeError("device lost")

        ledger = ExecutionTimingLedger(COST_LABEL_CATEGORIES, clock=clock,
                                       synchronize=sync)
        ledger.begin_envelope()
        ledger.start("client_training", round_id=4)
        clock.advance(3)
        partial = ledger.partial_snapshot()
        self.assertTrue(partial["tail_unmeasured"])
        self.assertIn("device lost", partial["capture_sample_error"])
        self.assertEqual(partial["label_seconds"]["client_training"], 0)

    def test_envelope_reports_real_gap_and_rejected_round_is_retained(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger({**LABELS, "envelope": UNCLASSIFIED_GAP}, clock=clock)
        with ledger.span("envelope"):
            with ledger.span("run", round_id=1):
                clock.advance(10)
            with ledger.span("wire", round_id=2):  # Rejected control: no completed metrics row.
                clock.advance(0.2)
            clock.advance(0.06)
        result = ledger.coverage("envelope")
        self.assertAlmostEqual(result["unclassified_seconds"], 0.06)
        self.assertFalse(result["passes"])
        self.assertIn(2, {event["round"] for event in ledger.snapshot()["events"]})
        restored = ExecutionTimingLedger.from_snapshot(
            json.loads(json.dumps(ledger.snapshot())), clock=clock)
        self.assertEqual(restored.coverage("envelope"), result)
        corrupt = copy.deepcopy(ledger.snapshot())
        corrupt["events"][0]["seconds"] += 1
        with self.assertRaisesRegex(ValueError, "event interval"):
            ExecutionTimingLedger.from_snapshot(corrupt, clock=clock)
        reordered = copy.deepcopy(ledger.snapshot())
        reordered["events"].reverse()
        with self.assertRaisesRegex(ValueError, "overlap or leave"):
            ExecutionTimingLedger.from_snapshot(reordered, clock=clock).coverage("envelope")

    def test_nested_intervals_charge_only_current_label_and_conserve_wall(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger(LABELS, clock=clock)
        with ledger.span("run"):
            clock.advance(2)
            with ledger.span("evaluation"):
                clock.advance(3)
                with ledger.span("wire"):
                    clock.advance(5)
                clock.advance(7)
            clock.advance(11)
        record = ledger.snapshot()
        self.assertEqual(record["label_seconds"], {
            "run": 13.0, "wire": 5.0, "evaluation": 10.0, "diagnostic": 0.0,
        })
        self.assertEqual(record["category_seconds"][NECESSARY_ALGORITHM], 18.0)
        self.assertEqual(record["category_seconds"][RESEARCH_OBSERVATION], 10.0)
        self.assertEqual(record["total_seconds"], 28.0)
        self.assertEqual(sum(record["category_seconds"].values()), clock.now)

    def test_exception_closes_span_and_optional_sync_wait_is_counted_once(self):
        clock = FakeClock()
        synchronizations = []

        def synchronize():
            synchronizations.append(clock.now)
            clock.advance(1)

        ledger = ExecutionTimingLedger(LABELS, clock=clock, synchronize=synchronize)
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            with ledger.span("run"):
                clock.advance(4)
                raise RuntimeError("worker failed")
        self.assertEqual(ledger.active_labels, ())
        self.assertEqual(len(synchronizations), 2)
        self.assertEqual(ledger.snapshot()["total_seconds"], 5.0)

    def test_diagnostic_and_research_nested_in_prefix_are_excluded_from_necessary(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger(LABELS, clock=clock)
        with ledger.span("run", bucket=DEPLOYED_PREFIX):
            clock.advance(2)
            with ledger.span("evaluation"):
                clock.advance(3)
            clock.advance(5)
        with ledger.span("diagnostic", bucket=DIAGNOSTIC_AFTER_BUDGET):
            clock.advance(7)
            with ledger.span("evaluation"):
                clock.advance(4)
        totals = ledger.snapshot()["category_seconds"]
        self.assertEqual(totals[NECESSARY_ALGORITHM], 7.0)
        self.assertEqual(totals[RESEARCH_OBSERVATION], 7.0)
        self.assertEqual(totals[POST_BUDGET_DIAGNOSTIC], 7.0)
        buckets = ledger.snapshot()["bucket_category_seconds"]
        self.assertEqual(buckets[DEPLOYED_PREFIX][NECESSARY_ALGORITHM], 7.0)
        self.assertEqual(buckets[DEPLOYED_PREFIX][RESEARCH_OBSERVATION], 3.0)
        self.assertEqual(buckets[DIAGNOSTIC_AFTER_BUDGET][RESEARCH_OBSERVATION], 4.0)
        self.assertEqual(sum(buckets[DIAGNOSTIC_AFTER_BUDGET].values()), 11.0)

    def test_snapshot_requires_closed_intervals_and_resume_preserves_totals(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger(LABELS, clock=clock)
        token = ledger.start("run")
        clock.advance(4)
        with self.assertRaises(UnclosedIntervalError):
            ledger.snapshot()
        ledger.stop(token)
        record = json.loads(json.dumps(ledger.snapshot()))
        clock.advance(1000)  # No active interval: downtime must not be charged.
        restored = ExecutionTimingLedger.from_snapshot(record, clock=clock)
        with restored.span("wire"):
            clock.advance(6)
        self.assertEqual(restored.snapshot()["label_seconds"]["run"], 4.0)
        self.assertEqual(restored.snapshot()["label_seconds"]["wire"], 6.0)
        self.assertEqual(restored.snapshot()["total_seconds"], 10.0)

    def test_unknown_label_and_out_of_order_stop_do_not_sample_clock(self):
        clock = FakeClock()
        samples = []

        def sampled_clock():
            samples.append(clock.now)
            return clock.now

        ledger = ExecutionTimingLedger(LABELS, clock=sampled_clock)
        with self.assertRaisesRegex(ValueError, "unknown timing label"):
            ledger.start("unclassified")
        with self.assertRaisesRegex(ValueError, "unknown timing bucket"):
            ledger.start("run", bucket="unclassified")
        self.assertEqual(samples, [])
        outer = ledger.start("run")
        inner = ledger.start("wire")
        before = len(samples)
        with self.assertRaisesRegex(RuntimeError, "LIFO"):
            ledger.stop(outer)
        self.assertEqual(len(samples), before)
        self.assertEqual(ledger.active_labels, ("run", "wire"))
        ledger.stop(inner)
        ledger.stop(outer)
        with self.assertRaisesRegex(ValueError, "unknown timing categories"):
            ExecutionTimingLedger({"x": "unclassified"}, clock=clock)

    def test_corrupt_snapshot_cannot_silently_change_accounting(self):
        clock = FakeClock()
        ledger = ExecutionTimingLedger(LABELS, clock=clock)
        with ledger.span("run"):
            clock.advance(2)
        record = ledger.snapshot()
        corrupted = copy.deepcopy(record)
        corrupted["category_seconds"][NECESSARY_ALGORITHM] = 0.0
        with self.assertRaisesRegex(ValueError, "category totals"):
            ExecutionTimingLedger.from_snapshot(corrupted, clock=clock)
        corrupted = copy.deepcopy(record)
        corrupted["bucket_label_seconds"][DEPLOYED_PREFIX]["run"] = -1.0
        with self.assertRaisesRegex(ValueError, "invalid seconds"):
            ExecutionTimingLedger.from_snapshot(corrupted, clock=clock)
        corrupted = copy.deepcopy(record)
        corrupted["label_seconds"]["run"] = 1.0
        with self.assertRaisesRegex(ValueError, "label totals"):
            ExecutionTimingLedger.from_snapshot(corrupted, clock=clock)


if __name__ == "__main__":
    unittest.main()
