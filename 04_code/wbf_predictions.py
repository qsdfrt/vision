"""Fuse COCO detection JSON files with lightweight weighted box fusion."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from vision_project.coco_utils import load_json, write_json


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def weighted_box(cluster: list[dict]) -> list[float]:
    total = sum(item["calibrated_score"] for item in cluster)
    return [
        sum(item["box"][coordinate] * item["calibrated_score"] for item in cluster) / total
        for coordinate in range(4)
    ]


def cluster_score(cluster: list[dict], source_weights: list[float], confidence_type: str) -> float:
    best_by_source: dict[int, dict] = {}
    for item in cluster:
        previous = best_by_source.get(item["source_index"])
        if previous is None or item["score"] > previous["score"]:
            best_by_source[item["source_index"]] = item

    if confidence_type == "max":
        return max(item["calibrated_score"] for item in best_by_source.values())
    numerator = sum(
        item["score"] * source_weights[source_index]
        for source_index, item in best_by_source.items()
    )
    if confidence_type == "average":
        denominator = sum(source_weights[source_index] for source_index in best_by_source)
    else:  # consensus: missing sources contribute a zero score.
        denominator = sum(source_weights)
    return numerator / max(denominator, 1e-9)


def fuse_wbf(
    sources: list[tuple[Path, float]],
    iou_threshold: float = 0.65,
    score_threshold: float = 0.001,
    output_score_threshold: float = 0.0,
    max_det: int = 300,
    confidence_type: str = "consensus",
) -> list[dict]:
    if confidence_type not in {"max", "average", "consensus"}:
        raise ValueError(f"Unsupported confidence type: {confidence_type}")
    source_weights = [weight for _, weight in sources]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for source_index, (source_path, weight) in enumerate(sources):
        for detection in load_json(source_path):
            score = float(detection["score"])
            calibrated_score = score * weight
            if calibrated_score < score_threshold:
                continue
            grouped[(int(detection["image_id"]), int(detection["category_id"]))].append({
                "box": xywh_to_xyxy(detection["bbox"]),
                "score": score,
                "calibrated_score": calibrated_score,
                "source_index": source_index,
            })

    by_image: dict[int, list[dict]] = defaultdict(list)
    for (image_id, category_id), detections in grouped.items():
        detections.sort(key=lambda item: item["calibrated_score"], reverse=True)
        clusters: list[list[dict]] = []
        representatives: list[list[float]] = []
        for detection in detections:
            best_index = -1
            best_iou = iou_threshold
            for index, representative in enumerate(representatives):
                overlap = box_iou(detection["box"], representative)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index < 0:
                clusters.append([detection])
                representatives.append(detection["box"][:])
            else:
                clusters[best_index].append(detection)
                representatives[best_index] = weighted_box(clusters[best_index])

        for cluster, representative in zip(clusters, representatives):
            score = cluster_score(cluster, source_weights, confidence_type)
            if score < output_score_threshold:
                continue
            by_image[image_id].append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [round(value, 2) for value in xyxy_to_xywh(representative)],
                "score": round(score, 8),
            })

    output: list[dict] = []
    for image_id in sorted(by_image):
        ranked = sorted(by_image[image_id], key=lambda item: item["score"], reverse=True)
        output.extend(ranked[:max_det])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs=2, action="append", metavar=("JSON", "WEIGHT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--output-score-threshold", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--confidence-type", choices=["max", "average", "consensus"], default="consensus")
    args = parser.parse_args()

    predictions = fuse_wbf(
        [(Path(path), float(weight)) for path, weight in args.source],
        iou_threshold=args.iou,
        score_threshold=args.score_threshold,
        output_score_threshold=args.output_score_threshold,
        max_det=args.max_det,
        confidence_type=args.confidence_type,
    )
    write_json(predictions, args.output)
    print(f"wrote {len(predictions)} WBF detections to {args.output}")


if __name__ == "__main__":
    main()
