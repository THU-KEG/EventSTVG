from role_stvg_eval.render import oracle_role_summary


def test_oracle_role_summary_uses_strict_thresholds():
    rows = [
        {"model": "M", "role": "A", "role_vIoU": 0.1},
        {"model": "M", "role": "A", "role_vIoU": 0.3},
    ]
    summary = oracle_role_summary(rows)[0]
    assert summary["N"] == 2
    assert summary["m_vIoU"] == 0.2
    assert summary["vIoU_gt_0.1"] == 0.5
    assert summary["vIoU_gt_0.3"] == 0.0

