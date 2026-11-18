import numpy as np
import pytest

from dental_agent.segmentation import contract, masks, metrics


def test_polygon_and_bbox_masks_agree():
    sq = masks.polygon_to_mask([(10, 10), (30, 10), (30, 30), (10, 30)], 64, 48)
    bb = masks.bbox_to_mask([10, 10, 20, 20], 64, 48)
    assert metrics.iou(sq, bb) > 0.8 and sq.shape == (48, 64)


def test_rle_roundtrip_including_edges():
    rng = np.random.default_rng(0)
    for m in (rng.random((17, 23)) > 0.5, np.zeros((5, 7), bool), np.ones((5, 7), bool)):
        assert np.array_equal(masks.rle_decode(masks.rle_encode(m)), m)
    with pytest.raises(ValueError):
        masks.rle_decode({"size": [2, 2], "counts": [1]})


def test_contract_validation():
    ok = contract.SegmentedFinding(1, 6, "Caries", polygon=[(1, 1), (5, 1), (5, 5)])
    contract.validate(ok, 10, 10)
    bad = [contract.SegmentedFinding(5, 6, "Caries", polygon=[(1, 1), (5, 1), (5, 5)]),
           contract.SegmentedFinding(1, 6, "Tumour", polygon=[(1, 1), (5, 1), (5, 5)]),
           contract.SegmentedFinding(1, 6, "Caries"),
           contract.SegmentedFinding(1, 6, "Caries", polygon=[(1, 1), (5, 1)]),
           contract.SegmentedFinding(1, 6, "Caries", polygon=[(1, 1), (50, 1), (5, 5)]),
           contract.SegmentedFinding(1, 6, "Caries", polygon=[(1, 1), (5, 1), (5, 5)], score=2.0)]
    for f in bad:
        with pytest.raises(contract.SegmentationContractError):
            contract.validate(f, 10, 10)


def test_matching_is_one_to_one_and_label_aware():
    a = masks.bbox_to_mask([0, 0, 10, 10], 40, 40)
    b = masks.bbox_to_mask([20, 20, 10, 10], 40, 40)
    gt = [{"mask": a, "fdi": 16, "diagnosis": "Caries"}, {"mask": b, "fdi": 38, "diagnosis": "Impacted Tooth"}]
    pred = [{"mask": a, "fdi": 16, "diagnosis": "Caries"}, {"mask": a, "fdi": 16, "diagnosis": "Caries"}, {"mask": b, "fdi": 38, "diagnosis": "Caries"}]
    r = metrics.match_masks(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 1)           # duplicate prediction not credited twice; wrong label not credited
    assert metrics.match_masks(gt, pred, require_label=False)["tp"] == 2
    p = metrics.pooled([r])
    assert p["f1"] == pytest.approx(2 / 5) and p["mean_matched_iou"] == 1.0


def test_empty_cases_are_zero_not_errors():
    assert metrics.match_masks([], [])["tp"] == 0
    assert metrics.pooled([metrics.match_masks([], [])])["f1"] == 0.0
    assert metrics.iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 0.0
