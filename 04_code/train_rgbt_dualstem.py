from __future__ import annotations

import argparse
from pathlib import Path

from train import KNOWN_WEIGHT_SOURCES, record_pretrained_weight
from vision_project.rgbt import RGBTDualStemTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("configs/rgbt_dataset.yaml"))
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="rgbt_dualstem_yolo11m_e30_i768")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    overrides = {
        "model": args.model,
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "fraction": args.fraction,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "deterministic": False,
        "amp": True,
        "cos_lr": True,
        "patience": args.patience,
        "plots": False,
        # Keep geometry synchronized across modalities. Horizontal flip is safe.
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "fliplr": 0.5,
        "close_mosaic": 0,
    }
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.lrf is not None:
        overrides["lrf"] = args.lrf
    if Path(args.model).name in KNOWN_WEIGHT_SOURCES:
        record_pretrained_weight(args.model)
    trainer = RGBTDualStemTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
