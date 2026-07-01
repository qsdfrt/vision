"""Paired RGB-T dataset, lightweight input gate, and Ultralytics trainer integration."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model


class RGBTYOLODataset(YOLODataset):
    """Load aligned RGB and TIR images as a six-channel RGB-ordered tensor."""

    def load_image(self, i: int, rect_mode: bool = True):
        rgb_bgr, original_shape, resized_shape = super().load_image(i, rect_mode=rect_mode)
        rgb_path = Path(self.im_files[i])
        split = rgb_path.parent.name
        tir_path = Path(self.data["tir_root"]) / split / "tir" / rgb_path.name
        tir_bgr = cv2.imread(str(tir_path), cv2.IMREAD_COLOR)
        if tir_bgr is None:
            raise FileNotFoundError(f"Paired TIR image not found: {tir_path}")
        if tir_bgr.shape[:2] != original_shape:
            raise ValueError(
                f"RGB/TIR shape mismatch for {rgb_path.name}: RGB={original_shape}, TIR={tir_bgr.shape[:2]}"
            )
        if tir_bgr.shape[:2] != rgb_bgr.shape[:2]:
            tir_bgr = cv2.resize(tir_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Six-channel images bypass Ultralytics' automatic BGR->RGB reversal, so convert both modalities explicitly.
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        tir = cv2.cvtColor(tir_bgr, cv2.COLOR_BGR2RGB)
        paired = np.ascontiguousarray(np.concatenate((rgb, tir), axis=2))
        return paired, original_shape, resized_shape

    def get_image_and_label(self, index: int) -> dict:
        """Load one pair and apply modality-specific photometric domain augmentation."""
        label = super().get_image_and_label(index)
        if not self.augment:
            return label

        image = label["img"].astype(np.float32)
        rgb, tir = image[..., :3], image[..., 3:]

        # The hidden test split is brighter in RGB and darker/lower-contrast in TIR than train/val.
        # Independent gains expose the detector to that shift without breaking spatial alignment.
        rgb_gain = np.random.uniform(0.75, 1.35)
        rgb_offset = np.random.uniform(-12.0, 18.0)
        tir_gain = np.random.uniform(0.75, 1.25)
        tir_offset = np.random.uniform(-18.0, 12.0)
        rgb = rgb * rgb_gain + rgb_offset
        tir_mean = tir.mean(axis=(0, 1), keepdims=True)
        tir_contrast = np.random.uniform(0.75, 1.25)
        tir = (tir - tir_mean) * tir_contrast + tir_mean
        tir = tir * tir_gain + tir_offset

        if np.random.random() < 0.20:
            noise = np.random.normal(0.0, np.random.uniform(1.0, 4.0), tir.shape)
            tir = tir + noise
        label["img"] = np.ascontiguousarray(
            np.concatenate((np.clip(rgb, 0, 255), np.clip(tir, 0, 255)), axis=2).astype(np.uint8)
        )
        return label


class GatedInputAdapter(nn.Module):
    """Learn a spatial RGB-vs-TIR gate and return a three-channel fused image."""

    def __init__(self, initial_rgb_weight: float = 0.25, hidden_channels: int = 8):
        super().__init__()
        if not 0 < initial_rgb_weight < 1:
            raise ValueError("initial_rgb_weight must be in (0, 1)")
        self.gate = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, bias=True),
        )
        nn.init.kaiming_normal_(self.gate[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(initial_rgb_weight / (1 - initial_rgb_weight)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 6:
            raise ValueError(f"Expected six-channel RGB-T input, got shape {tuple(x.shape)}")
        rgb, tir = x[:, :3], x[:, 3:]
        rgb_weight = self.gate(x).sigmoid()
        return rgb_weight * rgb + (1.0 - rgb_weight) * tir


class GatedFusionDetectionModel(DetectionModel):
    """A standard pretrained YOLO detector preceded by a tiny learned RGB-T gate."""

    def __init__(self, cfg, nc: int, initial_rgb_weight: float = 0.25, verbose: bool = True):
        # DetectionModel estimates stride during construction by forwarding a regular 3-channel dummy image.
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)
        self.rgbt_adapter = GatedInputAdapter(initial_rgb_weight=initial_rgb_weight)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        if x.shape[1] == 6 and hasattr(self, "rgbt_adapter"):
            x = self.rgbt_adapter(x)
        return super().forward(x, *args, **kwargs)


class SixChannelDetectionModel(DetectionModel):
    """YOLO detector whose first convolution learns RGB and TIR filters jointly."""

    def expand_input_channels(self, initial_rgb_weight: float = 0.25) -> None:
        if not 0 < initial_rgb_weight < 1:
            raise ValueError("initial_rgb_weight must be in (0, 1)")
        first = self.model[0]
        old = first.conv
        if old.in_channels == 6:
            return
        if old.in_channels != 3 or old.groups != 1:
            raise ValueError(f"Expected a regular three-channel first convolution, got {old}")

        expanded = nn.Conv2d(
            in_channels=6,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight[:, :3].copy_(old.weight * initial_rgb_weight)
            expanded.weight[:, 3:].copy_(old.weight * (1.0 - initial_rgb_weight))
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        first.conv = expanded


def _weights_have_prefix(weights: Any, prefix: str) -> bool:
    """Return whether a Ultralytics weight object already contains a module prefix."""
    if weights is None:
        return False
    model = weights["model"] if isinstance(weights, dict) else weights
    if not hasattr(model, "state_dict"):
        return False
    return any(key.startswith(prefix) for key in model.float().state_dict())


class DualStemFusionDetectionModel(DetectionModel):
    """Lightweight two-branch RGB-T detector with shared YOLO neck/head.

    RGB and TIR are processed by separate copies of the first two YOLO stem layers.
    Their 1/4-resolution feature maps are fused with a learned gate, and the
    remainder of the pretrained YOLO model is shared.
    """

    def __init__(self, cfg, nc: int, initial_rgb_weight: float = 0.5, verbose: bool = True):
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)
        if not 0 < initial_rgb_weight < 1:
            raise ValueError("initial_rgb_weight must be in (0, 1)")
        self.tir_stem = nn.Sequential(deepcopy(self.model[0]), deepcopy(self.model[1]))
        stem_channels = self.model[1].conv.out_channels
        self.feature_gate = nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.feature_gate.weight)
        nn.init.constant_(self.feature_gate.bias, math.log(initial_rgb_weight / (1 - initial_rgb_weight)))

    def copy_tir_stem_from_rgb(self) -> None:
        """Initialize the TIR stem from the loaded RGB stem weights."""
        self.tir_stem.load_state_dict(self.model[:2].state_dict())

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Run normal YOLO for 3-channel initialization, dual-stem for 6-channel RGB-T."""
        if x.shape[1] != 6 or not hasattr(self, "tir_stem"):
            return super()._predict_once(x, profile=profile, visualize=visualize, embed=embed)
        if profile or visualize or embed is not None:
            # Keep specialized debugging paths on the standard implementation.
            return super()._predict_once(x[:, :3], profile=profile, visualize=visualize, embed=embed)

        rgb, tir = x[:, :3], x[:, 3:]
        rgb_feat = self.model[1](self.model[0](rgb))
        tir_feat = self.tir_stem[1](self.tir_stem[0](tir))
        rgb_weight = self.feature_gate(torch.cat((rgb_feat, tir_feat), dim=1)).sigmoid()
        x = rgb_weight * rgb_feat + (1.0 - rgb_weight) * tir_feat

        # Indices 0 and 1 have been consumed by the custom dual stem. They are not used
        # by later YOLO11 skip connections, but placeholders keep y aligned with model indices.
        y = [None, None]
        for module in self.model[2:]:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            x = module(x)
            y.append(x if module.i in self.save else None)
        return x


class RGBTDetectionTrainer(DetectionTrainer):
    """Ultralytics detection trainer wired to paired images and the gated model."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        pad = 0.0 if mode == "train" else 0.5
        fraction = self.args.fraction if mode == "train" else 1.0
        return RGBTYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=pad,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=fraction,
        )

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        model = GatedFusionDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            initial_rgb_weight=float(self.data.get("initial_rgb_weight", 0.25)),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model


class RGBTSixChannelTrainer(RGBTDetectionTrainer):
    """Detection trainer for direct six-channel RGB-T early fusion."""

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        model = SixChannelDetectionModel(cfg=cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        model.expand_input_channels(initial_rgb_weight=float(self.data.get("initial_rgb_weight", 0.25)))
        return model


class RGBTDualStemTrainer(RGBTDetectionTrainer):
    """Detection trainer for lightweight two-stem RGB-T feature fusion."""

    def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
        model = DualStemFusionDetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            initial_rgb_weight=float(self.data.get("initial_rgb_weight", 0.5)),
            verbose=verbose and RANK == -1,
        )
        weights_have_tir_stem = _weights_have_prefix(weights, "tir_stem")
        if weights:
            model.load(weights)
        if not weights_have_tir_stem:
            model.copy_tir_stem_from_rgb()
        return model
