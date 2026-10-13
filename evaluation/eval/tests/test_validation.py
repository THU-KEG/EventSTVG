from role_stvg_eval.validation import INVALID, canonicalize_rows


BOX = [0, 0, 10, 10]


def test_duplicate_timestamp_invalidates_timestamp_without_last_row_wins():
    rows = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.0, "A": [20, 20, 30, 30]},
    ]
    view = canonicalize_rows(rows, [0.0], [0], ["A"])
    assert view.state(0, "A") == INVALID
    assert view.diagnostics["duplicate_timestamps"] == 1


def test_row_order_does_not_change_canonical_states():
    ordered = [
        {"time_sec": 0.0, "A": BOX},
        {"time_sec": 0.5, "A": None},
    ]
    forward = canonicalize_rows(ordered, [0.0, 0.5], [0, 1], ["A"])
    reverse = canonicalize_rows(
        list(reversed(ordered)), [0.0, 0.5], [0, 1], ["A"],
    )
    assert forward.states == reverse.states
    assert reverse.diagnostics["out_of_order_rows"] == 1


def test_off_grid_timestamp_is_ignored():
    view = canonicalize_rows(
        [{"time_sec": 0.25, "A": BOX}], [0.0, 0.5], [0, 1], ["A"],
    )
    assert view.diagnostics["ignored_extra_timestamps"] == 1
    assert all(view.state(index, "A").kind == "missing" for index in (0, 1))
