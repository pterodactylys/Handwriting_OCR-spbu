from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.tokenizer import CharacterTokenizer
from src.utils import resolve_path


@dataclass
class AugmentationConfig:
    enabled: bool = False
    rotation_deg: float = 2.0
    translate_px: int = 2
    blur_prob: float = 0.2
    blur_radius: float = 0.4
    contrast_min: float = 0.9
    contrast_max: float = 1.1
    noise_std: float = 0.02

    @classmethod
    def from_dict(cls, payload: dict | None, enabled: bool = False) -> "AugmentationConfig":
        payload = payload or {}
        return cls(enabled=enabled, **payload)


def apply_light_augmentations(image: Image.Image, config: AugmentationConfig) -> Image.Image:
    if not config.enabled:
        return image

    angle = random.uniform(-config.rotation_deg, config.rotation_deg)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )

    translate_x = random.uniform(-config.translate_px, config.translate_px)
    translate_y = random.uniform(-config.translate_px, config.translate_px)
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, translate_x, 0.0, 1.0, translate_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )

    contrast_factor = random.uniform(config.contrast_min, config.contrast_max)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    if random.random() < config.blur_prob and config.blur_radius > 0.0:
        radius = random.uniform(0.0, config.blur_radius)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    return image


def resize_preserving_aspect(image: Image.Image, img_height: int) -> Image.Image:
    original_width, original_height = image.size
    scaled_width = max(1, round(original_width * (img_height / original_height)))
    return image.resize((scaled_width, img_height), resample=Image.Resampling.BICUBIC)


def pil_to_normalized_tensor(
    image: Image.Image,
    *,
    add_noise_std: float = 0.0,
) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if add_noise_std > 0.0:
        noise = torch.randn_like(tensor) * add_noise_std
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return (tensor - 0.5) / 0.5


def preprocess_line_image(
    image_path: str | Path,
    *,
    img_height: int,
    augmentation_config: AugmentationConfig | None = None,
    apply_augmentations: bool = False,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    resolved_path = resolve_path(image_path)
    with Image.open(resolved_path) as handle:
        image = handle.convert("L")

    original_size = image.size
    augmentation_config = augmentation_config or AugmentationConfig()
    if apply_augmentations:
        image = apply_light_augmentations(image, augmentation_config)

    resized = resize_preserving_aspect(image, img_height=img_height)
    tensor = pil_to_normalized_tensor(
        resized,
        add_noise_std=augmentation_config.noise_std if apply_augmentations else 0.0,
    )
    return tensor, original_size, resized.size


class IAMLineDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharacterTokenizer,
        *,
        img_height: int = 32,
        train: bool = False,
        augmentations: AugmentationConfig | None = None,
        max_samples: int = 0,
    ) -> None:
        self.manifest_path = resolve_path(manifest_path)
        self.tokenizer = tokenizer
        self.img_height = img_height
        self.train = train
        self.augmentations = augmentations or AugmentationConfig()

        frame = pd.read_csv(self.manifest_path)
        if max_samples > 0:
            frame = frame.iloc[:max_samples].copy()
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = str(row["image_path"])
        text = str(row["text"])
        image_tensor, original_size, resized_size = preprocess_line_image(
            image_path,
            img_height=self.img_height,
            augmentation_config=self.augmentations,
            apply_augmentations=self.train and self.augmentations.enabled,
        )
        target = self.tokenizer.text_to_tensor(text)
        return {
            "image": image_tensor,
            "target": target,
            "text": text,
            "image_path": image_path,
            "original_size": original_size,
            "resized_size": resized_size,
            "image_width": resized_size[0],
        }


def ctc_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("Received an empty batch.")

    batch_size = len(batch)
    image_height = batch[0]["image"].shape[-2]  # type: ignore[index]
    max_width = max(int(sample["image_width"]) for sample in batch)

    images = torch.full(
        (batch_size, 1, image_height, max_width),
        fill_value=-1.0,
        dtype=batch[0]["image"].dtype,  # type: ignore[index]
    )
    targets = []
    target_lengths = []
    texts: list[str] = []
    image_paths: list[str] = []
    image_widths = []
    original_sizes: list[tuple[int, int]] = []
    resized_sizes: list[tuple[int, int]] = []

    for batch_index, sample in enumerate(batch):
        image = sample["image"]  # type: ignore[assignment]
        width = int(sample["image_width"])
        images[batch_index, :, :, :width] = image
        targets.append(sample["target"])
        target_lengths.append(len(sample["target"]))  # type: ignore[arg-type]
        texts.append(str(sample["text"]))
        image_paths.append(str(sample["image_path"]))
        image_widths.append(width)
        original_sizes.append(tuple(sample["original_size"]))  # type: ignore[arg-type]
        resized_sizes.append(tuple(sample["resized_size"]))  # type: ignore[arg-type]

    return {
        "images": images,
        "targets": torch.cat(targets, dim=0),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "image_paths": image_paths,
        "image_widths": torch.tensor(image_widths, dtype=torch.long),
        "original_sizes": original_sizes,
        "resized_sizes": resized_sizes,
    }
