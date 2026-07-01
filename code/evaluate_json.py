from __future__ import annotations

import argparse
from pathlib import Path

from vision_project.coco_eval import evaluate_coco_results
from vision_project.coco_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("data/rgbt_vehicle/val/val.json"))
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    metrics = evaluate_coco_results(args.annotations, args.predictions)
    output = args.metrics or args.predictions.with_name(f"{args.predictions.stem}_metrics.json")
    write_json(metrics, output, indent=2)
    print(f"wrote metrics to {output}")


if __name__ == "__main__":
    main()

