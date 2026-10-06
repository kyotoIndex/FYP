"""Independent CPU ordering check for the mutable conditional replay adapter."""
import pytest

import pcla_research_v7.cohort_runtime as cohort_runtime
import pcla_research_v7.conditional_replay as conditional_replay
import pcla_research_v7.runtime as runtime
from tests_pcla_v7.test_conditional_replay_integration import (
    IDENTITIES, _book, _config, _inputs, _job, _support,
)


@pytest.mark.parametrize("representation", ["D", "T"])
def test_current_coordinate_path_precedes_replay_mask_and_actual_wire(
        tmp_path, monkeypatch, representation):
    """A donor count may replace only the mask, after the replay's own parent."""
    order = []
    if representation == "D":
        original = runtime.refresh_coordinates

        def coordinate(*args, **kwargs):
            order.append(("coordinate", 2))
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime, "refresh_coordinates", coordinate)
    else:
        original = cohort_runtime.prepare_cohort

        def coordinate(*args, **kwargs):
            order.append(("coordinate", kwargs["round_id"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(cohort_runtime, "prepare_cohort", coordinate)

    original_preview = runtime.RawTensorWire.preview

    def actual_wire(self, *args, **kwargs):
        order.append(("wire_preview", kwargs["round_id"]))
        return original_preview(self, *args, **kwargs)

    def quota_masks(**kwargs):
        order.append(("quota_masks", kwargs["round_id"]))
        return _support(**kwargs)

    monkeypatch.setattr(runtime.RawTensorWire, "preview", actual_wire)
    monkeypatch.setattr(conditional_replay, "replay_support", quota_masks)
    model, clients, capacities, evaluation = _inputs()
    rows = runtime.run_experiment(model, clients, evaluation, capacities,
        _job(representation), _config(), tmp_path / representation,
        {"evaluation_role": "synthetic"}, stop_after=2,
        replay_book=_book(representation), replay_expected_identities=IDENTITIES)
    assert order.index(("coordinate", 2)) < order.index(("quota_masks", 2)) < order.index(("wire_preview", 2))
    assert rows[1]["quota_replay"]["coordinate_path"] == representation
    assert rows[1]["deployed_bytes"] > rows[0]["deployed_bytes"] > 0
