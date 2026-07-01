from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: Path, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            allow_nan=False,
        )


def xyxy_to_xywh(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = map(float, box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    w, h = x2 - x1, y2 - y1
    if not all(math.isfinite(value) for value in (x1, y1, w, h)) or w <= 0 or h <= 0:
        return None
    return [x1, y1, w, h]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
