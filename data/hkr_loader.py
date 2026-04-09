from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _resolve_image_path(img_dir: Path, sample_name: str) -> Path | None:
    """Find an image file by sample name using common extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = img_dir / f"{sample_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def validate_hkr_words_dataset(dataset_root: str | Path) -> Dict[str, int]:
    """Validate required HKR words dataset folders and return basic counts."""
    root = Path(dataset_root)
    ann_dir = root / "ann"
    img_dir = root / "img"

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not ann_dir.exists() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    ann_count = sum(1 for p in ann_dir.glob("*.json") if p.is_file())
    img_count = sum(1 for p in img_dir.iterdir() if p.is_file())

    if ann_count == 0:
        raise ValueError(f"No annotation files found in: {ann_dir}")
    if img_count == 0:
        raise ValueError(f"No images found in: {img_dir}")

    return {
        "ann_count": ann_count,
        "img_count": img_count,
    }


def parse_hkr_words_dataset(
    dataset_root: str | Path,
    skip_unmoderated: bool = True,
    strip_text: bool = True,
) -> Tuple[List[dict], int]:
    """
    Parse HKR words dataset into a list of samples used by the CRNN pipeline.

    Returns:
        samples: list of dicts with keys compatible with the training notebook
        missing_images: number of annotation records without an image
    """
    root = Path(dataset_root)
    ann_dir = root / "ann"
    img_dir = root / "img"

    samples: List[dict] = []
    missing_images = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        with ann_path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        sample_name = str(record.get("name", "")).strip()
        text = str(record.get("description", ""))

        if strip_text:
            text = text.strip()

        if not sample_name:
            continue
        if not text:
            continue

        if skip_unmoderated:
            moderation = record.get("moderation", {})
            is_moderated = int(moderation.get("isModerated", 0)) == 1
            if not is_moderated:
                continue

        image_path = _resolve_image_path(img_dir, sample_name)
        if image_path is None:
            missing_images += 1
            continue

        size = record.get("size", {})
        width = int(size.get("width", 0) or 0)
        height = int(size.get("height", 0) or 0)
        bbox = (0, 0, width, height)

        samples.append(
            {
                "line_id": sample_name,
                "transcription": text,
                "image_path": image_path,
                "bbox": bbox,
            }
        )

    return samples, missing_images
