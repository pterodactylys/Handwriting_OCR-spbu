from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class ImageConfig:
    height: int = 64
    width: int = 512
    pad_value: int = 255


def resize_with_aspect_and_pad(image: Image.Image, cfg: ImageConfig) -> Image.Image:
    """Resize image preserving aspect ratio and pad to fixed canvas size."""
    image = image.convert("L")
    src_w, src_h = image.size

    scale = min(cfg.width / max(src_w, 1), cfg.height / max(src_h, 1))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    resized = image.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (cfg.width, cfg.height), color=cfg.pad_value)

    offset_x = (cfg.width - dst_w) // 2
    offset_y = (cfg.height - dst_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL grayscale image to float tensor in [0, 1] with shape [1, H, W]."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_tensor(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    return (tensor - mean) / std


def preprocess_image(image: Image.Image, cfg: ImageConfig) -> torch.Tensor:
    image = resize_with_aspect_and_pad(image, cfg)
    tensor = pil_to_tensor(image)
    return normalize_tensor(tensor)


def make_input_shape(cfg: ImageConfig) -> Tuple[int, int, int]:
    return (1, cfg.height, cfg.width)
