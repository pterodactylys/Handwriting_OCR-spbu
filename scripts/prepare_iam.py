from __future__ import annotations

import argparse
import html
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable | None = None, **_: object) -> Iterable | None:
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_DIR = PROJECT_ROOT / "data" / "xml"
DEFAULT_LINES_DIR = PROJECT_ROOT / "data" / "lines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class LineSample:
    image_path: str
    text: str
    writer_id: str
    form_id: str
    line_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IAM line-level manifests.")
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--lines-dir", type=Path, default=DEFAULT_LINES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--include-segmentation-errors",
        action="store_true",
        help="Keep XML lines with segmentation='err'. By default they are dropped.",
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Do not open images with Pillow during manifest generation.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\t", " ").strip()
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split())


def verify_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def collect_samples(
    xml_dir: Path,
    lines_dir: Path,
    *,
    include_segmentation_errors: bool,
    verify_images: bool,
) -> tuple[list[LineSample], Counter[str]]:
    dropped = Counter()
    samples: list[LineSample] = []
    xml_files = sorted(xml_dir.glob("*.xml"))

    for xml_path in tqdm(xml_files, desc="Parsing XML files"):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            dropped["xml_parse_error"] += 1
            continue

        form_id = root.get("id")
        writer_id = root.get("writer-id")
        if not form_id or not writer_id:
            dropped["missing_form_metadata"] += 1
            continue

        form_lines_dir = lines_dir / form_id[:3] / form_id
        for line in root.findall("./handwritten-part/line"):
            segmentation = (line.get("segmentation") or "").strip().lower()
            if segmentation == "err" and not include_segmentation_errors:
                dropped["segmentation_err"] += 1
                continue

            line_id = line.get("id")
            raw_text = line.get("text")
            if not line_id:
                dropped["missing_line_id"] += 1
                continue
            if raw_text is None:
                dropped["missing_text"] += 1
                continue

            text = normalize_text(raw_text)
            if not text:
                dropped["empty_text_after_normalization"] += 1
                continue

            image_path = form_lines_dir / f"{line_id}.png"
            if not image_path.exists():
                dropped["missing_image"] += 1
                continue
            if verify_images and not verify_image(image_path):
                dropped["invalid_image"] += 1
                continue

            samples.append(
                LineSample(
                    image_path=relative_to_project(image_path),
                    text=text,
                    writer_id=writer_id,
                    form_id=form_id,
                    line_id=line_id,
                )
            )

    return samples, dropped


def split_writers(
    writer_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    unique_writers = sorted(set(writer_ids))
    shuffled = unique_writers[:]
    random.Random(seed).shuffle(shuffled)

    writer_count = len(shuffled)
    train_count = max(1, int(writer_count * train_ratio))
    val_count = max(1, int(writer_count * val_ratio))

    if train_count + val_count >= writer_count:
        val_count = max(1, writer_count - train_count - 1)
    test_count = writer_count - train_count - val_count
    if test_count <= 0:
        raise ValueError("Not enough writers to create a non-empty test split.")

    split_map: dict[str, str] = {}
    for writer_id in shuffled[:train_count]:
        split_map[writer_id] = "train"
    for writer_id in shuffled[train_count : train_count + val_count]:
        split_map[writer_id] = "val"
    for writer_id in shuffled[train_count + val_count :]:
        split_map[writer_id] = "test"
    return split_map


def ensure_train_character_coverage(
    samples: list[LineSample],
    split_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    chars_by_writer: dict[str, set[str]] = defaultdict(set)
    split_sizes = Counter(split_map.values())
    for sample in samples:
        chars_by_writer[sample.writer_id].update(sample.text)

    all_chars = set().union(*chars_by_writer.values())
    train_chars = set().union(
        *(chars_by_writer[writer_id] for writer_id, split in split_map.items() if split == "train")
    )
    moved_writers: list[str] = []

    while missing_chars := sorted(all_chars - train_chars):
        candidates: list[tuple[int, str, str]] = []
        for writer_id, split_name in split_map.items():
            if split_name == "train":
                continue
            if split_sizes[split_name] <= 1:
                continue
            coverage = len(chars_by_writer[writer_id].intersection(missing_chars))
            if coverage > 0:
                candidates.append((coverage, split_name, writer_id))

        if not candidates:
            raise RuntimeError(
                "Unable to ensure that the train split covers the full character alphabet."
            )

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, old_split, chosen_writer = candidates[0]
        split_map[chosen_writer] = "train"
        split_sizes["train"] += 1
        split_sizes[old_split] -= 1
        train_chars.update(chars_by_writer[chosen_writer])
        moved_writers.append(chosen_writer)

    return split_map, moved_writers


def build_split_frames(
    samples: list[LineSample],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    split_by_writer = split_writers(
        [sample.writer_id for sample in samples],
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split_by_writer, moved_writers = ensure_train_character_coverage(samples, split_by_writer)

    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        split_name = split_by_writer[sample.writer_id]
        rows_by_split[split_name].append(
            {
                "image_path": sample.image_path,
                "text": sample.text,
            }
        )

    frames = {
        split_name: pd.DataFrame(rows, columns=["image_path", "text"]).sort_values("image_path")
        for split_name, rows in rows_by_split.items()
    }
    return frames, moved_writers


def describe_split(name: str, frame: pd.DataFrame) -> None:
    lengths = frame["text"].astype(str).map(len)
    unique_chars = sorted({char for text in frame["text"].astype(str) for char in text})
    print(
        f"{name:>5}: {len(frame):5d} samples | "
        f"avg_len={lengths.mean():.2f} | min_len={lengths.min()} | max_len={lengths.max()} | "
        f"unique_chars={len(unique_chars)}"
    )
    print(f"       chars: {''.join(unique_chars)}")


def save_manifests(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in frames.items():
        output_path = output_dir / f"iam_manifest_{split_name}.csv"
        frame.to_csv(output_path, index=False)
        print(f"Saved {split_name} manifest to {output_path}")


def main() -> None:
    args = parse_args()
    xml_dir = args.xml_dir if args.xml_dir.is_absolute() else PROJECT_ROOT / args.xml_dir
    lines_dir = args.lines_dir if args.lines_dir.is_absolute() else PROJECT_ROOT / args.lines_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    samples, dropped = collect_samples(
        xml_dir=xml_dir,
        lines_dir=lines_dir,
        include_segmentation_errors=args.include_segmentation_errors,
        verify_images=not args.skip_image_verification,
    )
    frames, moved_writers = build_split_frames(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    total_kept = sum(len(frame) for frame in frames.values())
    print(f"Total kept samples: {total_kept}")
    print(f"Total dropped samples: {sum(dropped.values())}")
    print(f"Writers moved into train to preserve full alphabet coverage: {len(moved_writers)}")
    for reason, count in dropped.most_common():
        print(f"  - {reason}: {count}")

    for split_name in ("train", "val", "test"):
        if split_name not in frames:
            raise RuntimeError(f"Split '{split_name}' is empty.")
        describe_split(split_name, frames[split_name])

    save_manifests(frames, output_dir)


if __name__ == "__main__":
    main()
