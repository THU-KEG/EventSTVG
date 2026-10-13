from role_stvg_eval.render import _split_latex


def scores():
    row = {"model": "M"}
    for prefix in ("tIoU", "vIoU", "vIoU_Int"):
        row[f"m_{prefix}"] = 0.1
        for threshold in (0, 0.1, 0.3, 0.5):
            row[f"{prefix}_gt_{threshold:g}"] = 0.1
    return [row]


def test_tg_and_sg_latex_are_separated_and_grouped():
    tg = _split_latex(scores(), "tg")
    sg = _split_latex(scores(), "sg")
    stvg = _split_latex(scores(), "stvg")
    viou_int = _split_latex(scores(), "viou_int")
    assert "m-tIoU" in tg and "m-vIoU" not in tg
    assert "m-vIoU" in sg and "m-vIoU-Int" in sg and "m-tIoU" not in sg
    assert "\\multicolumn{5}{c}{Joint STVG}" in sg
    assert "m-vIoU" in stvg and "m-vIoU-Int" not in stvg
    assert "m-vIoU-Int" in viou_int and "m-vIoU &" not in viou_int
