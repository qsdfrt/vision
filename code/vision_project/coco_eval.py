from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
    "AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def evaluate_coco_results(annotations: Path, predictions: Path, *, summarize: bool = True) -> dict:
    coco_gt = COCO(str(annotations.resolve()))
    coco_dt = coco_gt.loadRes(str(predictions.resolve()))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    if summarize:
        evaluator.summarize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.summarize()

    metrics = {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}
    precision = evaluator.eval["precision"]
    categories = sorted(coco_gt.dataset["categories"], key=lambda item: item["id"])
    for category_index, category in enumerate(categories):
        values = precision[:, :, category_index, 0, -1]
        values = values[values > -1]
        metrics[f"AP_{category['name']}"] = float(np.mean(values)) if values.size else None
    metrics["prediction_count"] = len(coco_dt.dataset["annotations"])
    return metrics
