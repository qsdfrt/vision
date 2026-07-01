"""Run paired RGB-T inference with simple flip TTA and unflip detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import nms, ops

from vision_project.coco_eval import evaluate_coco_results
from vision_project.coco_utils import write_json, xyxy_to_xywh
from vision_project.rgbt import (
    DualStemFusionDetectionModel,
    GatedFusionDetectionModel,
    RGBTYOLODataset,
    SixChannelDetectionModel,
)


def build_dataset(data: dict, images: Path, imgsz: int, batch: int) -> RGBTYOLODataset:
    return RGBTYOLODataset(
        img_path=str(images.resolve()),
        imgsz=imgsz,
        batch_size=batch,
        augment=False,
        hyp=get_cfg(),
        rect=True,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="predict-tta: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )


def apply_tta(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return images
    if mode == "hflip":
        return torch.flip(images, dims=[3])
    if mode == "vflip":
        return torch.flip(images, dims=[2])
    if mode == "hvflip":
        return torch.flip(images, dims=[2, 3])
    raise ValueError(f"Unsupported TTA mode: {mode}")


def unflip_boxes(detection: torch.Tensor, mode: str, padded_height: int, padded_width: int) -> torch.Tensor:
    if mode == "none":
        return detection
    detection = detection.clone()
    if mode in {"hflip", "hvflip"}:
        x1 = detection[:, 0].clone()
        x2 = detection[:, 2].clone()
        detection[:, 0] = padded_width - x2
        detection[:, 2] = padded_width - x1
    if mode in {"vflip", "hvflip"}:
        y1 = detection[:, 1].clone()
        y2 = detection[:, 3].clone()
        detection[:, 1] = padded_height - y2
        detection[:, 3] = padded_height - y1
    return detection


def predict_tta(
    weights: Path,
    data_path: Path,
    images: Path,
    output: Path,
    imgsz: int,
    batch_size: int,
    workers: int,
    device_name: str,
    conf: float,
    iou: float,
    max_det: int,
    gate_bias_delta: float,
    tta: str,
) -> None:
    with data_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["tir_root"] = str(Path(data["tir_root"]).resolve())

    dataset = build_dataset(data, images, imgsz, batch_size)
    loader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False, rank=-1)
    device = torch.device(f"cuda:{device_name}" if device_name.isdigit() and torch.cuda.is_available() else device_name)
    model, _ = load_checkpoint(str(weights.resolve()), device=device, fuse=False)
    if not isinstance(model, (GatedFusionDetectionModel, SixChannelDetectionModel, DualStemFusionDetectionModel)):
        raise TypeError(f"Expected an RGB-T detection model, got {type(model).__name__}")
    if gate_bias_delta and not isinstance(model, GatedFusionDetectionModel):
        raise ValueError("--gate-bias-delta is only valid for GatedFusionDetectionModel")
    if gate_bias_delta:
        with torch.no_grad():
            model.rgbt_adapter.gate[-1].bias.add_(gate_bias_delta)
        print(f"applied gate bias delta: {gate_bias_delta:+.3f}")
    model.eval()

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            images_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            tta_images = apply_tta(images_tensor, tta)
            raw = model(tta_images)
            detections = nms.non_max_suppression(
                raw,
                conf_thres=conf,
                iou_thres=iou,
                nc=5,
                multi_label=True,
                max_det=max_det,
            )
            padded_height, padded_width = images_tensor.shape[2], images_tensor.shape[3]
            for index, detection in enumerate(detections):
                if not len(detection):
                    continue
                detection = unflip_boxes(detection, tta, padded_height, padded_width)
                boxes = ops.scale_boxes(
                    images_tensor.shape[2:],
                    detection[:, :4].clone(),
                    batch["ori_shape"][index],
                    ratio_pad=batch["ratio_pad"][index],
                )
                image_path = Path(batch["im_file"][index])
                image_id = int(image_path.stem)
                height, width = batch["ori_shape"][index]
                for box, score, class_index in zip(
                    boxes.cpu().tolist(), detection[:, 4].cpu().tolist(), detection[:, 5].cpu().tolist()
                ):
                    bbox = xyxy_to_xywh(box, int(width), int(height))
                    if bbox is not None:
                        predictions.append({
                            "image_id": image_id,
                            "category_id": int(class_index) + 1,
                            "bbox": [round(value, 1) for value in bbox],
                            "score": round(float(score), 6),
                        })
    write_json(predictions, output)
    print(f"wrote {len(predictions)} {tta} paired detections to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/rgbt_dataset.yaml"))
    parser.add_argument("--images", type=Path, default=Path("data/yolo_rgb/images/test"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission_rgbt_tta.json"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--gate-bias-delta", type=float, default=0.0)
    parser.add_argument("--tta", choices=["none", "hflip", "vflip", "hvflip"], required=True)
    args = parser.parse_args()

    predict_tta(
        args.weights, args.data, args.images, args.output, args.imgsz,
        args.batch, args.workers, args.device, args.conf, args.iou, args.max_det,
        args.gate_bias_delta, args.tta,
    )
    if args.annotations:
        metrics = evaluate_coco_results(args.annotations, args.output)
        metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
        write_json(metrics, metrics_path, indent=2)
        print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
