from role_stvg_eval.boxes import box_iou, valid_box, valid_normalized_box


def test_box_iou_identity_and_disjoint():
    box = [0, 0, 10, 10]
    assert box_iou(box, box) == 1.0
    assert box_iou(box, [20, 20, 30, 30]) == 0.0


def test_strict_scoring_rejects_malformed_boxes():
    invalid = [
        [-1, 0, 10, 10],
        [0, 0, 1001, 10],
        [0, 0, 10],
        [0, 0, 0, 10],
        [0, 10, 10, 10],
        [False, 0, 10, 10],
        [0, 0, float("nan"), 10],
        [0, 0, float("inf"), 10],
    ]
    assert all(not valid_box(box) for box in invalid)
    assert all(not valid_normalized_box(box) for box in invalid)
    assert valid_box([0, 0, 1000, 1000])
