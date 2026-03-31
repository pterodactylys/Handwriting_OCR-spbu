"""Utilities for loading and exporting IAM line-level samples from raw/.

The module provides:
1) `IAMLinesIndex` for convenient filtering and export.
2) `IAMLinesTorchDataset` for training/inference pipelines.
3) CLI for quick stats and manifest export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


VALID_SPLITS = {"train", "val", "test", "all"}


@dataclass(frozen=True)
class IAMLineSample:
	line_id: str
	form_id: str
	writer_id: str
	status: str
	text: str
	image_path: str


def _resolve_raw_root(root_dir: str | Path) -> Path:
	"""Resolve path to raw/ directory (strictly through raw)."""
	root = Path(root_dir).expanduser().resolve()
	raw = root if root.name == "raw" else root / "raw"

	lines_txt = raw / "ascii" / "lines.txt"
	lines_dir = raw / "lines"
	if not lines_txt.exists() or not lines_dir.exists():
		raise FileNotFoundError(
			f"IAM raw directory is invalid: expected '{lines_txt}' and '{lines_dir}'."
		)
	return raw


def _split_bucket(key: str) -> int:
	"""Deterministic bucket in [0, 9] for stable train/val/test partition."""
	return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10


def _is_in_split(form_id: str, split: str) -> bool:
	if split == "all":
		return True
	bucket = _split_bucket(form_id)
	if split == "train":
		return bucket < 8
	if split == "val":
		return bucket == 8
	return bucket == 9


def _line_image_path(raw_root: Path, line_id: str) -> Path:
	# IAM layout: raw/lines/<writer_id>/<form_id>/<line_id>.png
	parts = line_id.split("-")
	writer_id = parts[0]
	form_id = "-".join(parts[:2])
	return raw_root / "lines" / writer_id / form_id / f"{line_id}.png"


class IAMLinesIndex:
	"""In-memory index over IAM line-level dataset."""

	def __init__(self, raw_root: str | Path, samples: Sequence[IAMLineSample]):
		self.raw_root = Path(raw_root)
		self.samples = list(samples)

	@classmethod
	def from_raw(
		cls,
		root_dir: str | Path,
		split: str = "train",
		include_err: bool = False,
		require_existing_images: bool = True,
		limit: Optional[int] = None,
	) -> "IAMLinesIndex":
		if split not in VALID_SPLITS:
			raise ValueError(f"split must be one of: {sorted(VALID_SPLITS)}")

		raw_root = _resolve_raw_root(root_dir)
		lines_txt = raw_root / "ascii" / "lines.txt"

		samples: List[IAMLineSample] = []
		with lines_txt.open("r", encoding="utf-8", errors="ignore") as file_obj:
			for raw_line in file_obj:
				line = raw_line.strip()
				if not line or line.startswith("#"):
					continue

				# Field #9 contains the full transcript and can include spaces.
				parts = line.split(maxsplit=8)
				if len(parts) < 9:
					continue

				line_id, status, transcript_raw = parts[0], parts[1], parts[8]
				if status == "err" and not include_err:
					continue

				id_parts = line_id.split("-")
				writer_id = id_parts[0]
				form_id = "-".join(id_parts[:2])

				if not _is_in_split(form_id, split):
					continue

				image_path = _line_image_path(raw_root, line_id)
				if require_existing_images and not image_path.exists():
					continue

				text = " ".join(transcript_raw.replace("|", " ").split())
				samples.append(
					IAMLineSample(
						line_id=line_id,
						form_id=form_id,
						writer_id=writer_id,
						status=status,
						text=text,
						image_path=str(image_path),
					)
				)

				if limit is not None and len(samples) >= limit:
					break

		return cls(raw_root=raw_root, samples=samples)

	def __len__(self) -> int:
		return len(self.samples)

	def filter(
		self,
		statuses: Optional[Iterable[str]] = None,
		writer_ids: Optional[Iterable[str]] = None,
		text_contains: Optional[str] = None,
	) -> "IAMLinesIndex":
		statuses_set = set(statuses) if statuses is not None else None
		writer_ids_set = set(writer_ids) if writer_ids is not None else None
		text_query = text_contains.lower() if text_contains else None

		filtered: List[IAMLineSample] = []
		for sample in self.samples:
			if statuses_set is not None and sample.status not in statuses_set:
				continue
			if writer_ids_set is not None and sample.writer_id not in writer_ids_set:
				continue
			if text_query is not None and text_query not in sample.text.lower():
				continue
			filtered.append(sample)

		return IAMLinesIndex(raw_root=self.raw_root, samples=filtered)

	def to_records(self) -> List[dict]:
		return [asdict(sample) for sample in self.samples]

	def save_manifest(self, output_path: str | Path, fmt: str = "jsonl") -> Path:
		output = Path(output_path).expanduser().resolve()
		output.parent.mkdir(parents=True, exist_ok=True)

		if fmt == "jsonl":
			with output.open("w", encoding="utf-8") as file_obj:
				for sample in self.samples:
					file_obj.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
			return output

		if fmt == "csv":
			with output.open("w", newline="", encoding="utf-8") as file_obj:
				writer = csv.DictWriter(
					file_obj,
					fieldnames=["line_id", "form_id", "writer_id", "status", "text", "image_path"],
				)
				writer.writeheader()
				for sample in self.samples:
					writer.writerow(asdict(sample))
			return output

		raise ValueError("fmt must be either 'jsonl' or 'csv'")

	def export_subset(self, output_dir: str | Path, copy_images: bool = True) -> Path:
		"""Export a compact subset with manifest + optional copied images."""
		out_dir = Path(output_dir).expanduser().resolve()
		images_dir = out_dir / "images"
		out_dir.mkdir(parents=True, exist_ok=True)
		if copy_images:
			images_dir.mkdir(parents=True, exist_ok=True)

		exported_records = []
		for sample in self.samples:
			sample_dict = asdict(sample)
			if copy_images:
				src = Path(sample.image_path)
				dst = images_dir / src.name
				shutil.copy2(src, dst)
				sample_dict["image_path"] = str(dst)
			exported_records.append(sample_dict)

		manifest = out_dir / "manifest.jsonl"
		with manifest.open("w", encoding="utf-8") as file_obj:
			for row in exported_records:
				file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
		return manifest


class IAMLinesTorchDataset(Dataset):
	"""PyTorch dataset wrapper over IAMLinesIndex."""

	def __init__(
		self,
		index: IAMLinesIndex,
		image_transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
	):
		self.index = index
		self.image_transform = image_transform
		self.target_transform = target_transform

	def __len__(self) -> int:
		return len(self.index)

	def __getitem__(self, idx: int):
		sample = self.index.samples[idx]
		image = Image.open(sample.image_path).convert("L")
		text = sample.text

		if self.image_transform is not None:
			image = self.image_transform(image)
		if self.target_transform is not None:
			text = self.target_transform(text)

		return image, text


def _build_cli_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="IAM raw dataset loader and exporter")
	parser.add_argument("--root_dir", type=str, required=True, help="Path to project root or raw/")
	parser.add_argument("--split", type=str, default="train", choices=sorted(VALID_SPLITS))
	parser.add_argument("--include_err", action="store_true", help="Include samples with status='err'")
	parser.add_argument("--limit", type=int, default=None, help="Limit number of loaded samples")
	parser.add_argument("--manifest", type=str, default=None, help="Output manifest path (jsonl/csv)")
	parser.add_argument("--manifest_format", type=str, default="jsonl", choices=["jsonl", "csv"])
	parser.add_argument("--export_dir", type=str, default=None, help="Export subset into this folder")
	parser.add_argument("--copy_images", action="store_true", help="Copy images when exporting subset")
	parser.add_argument("--preview", type=int, default=3, help="How many rows to print")
	return parser


def main() -> None:
	args = _build_cli_parser().parse_args()

	index = IAMLinesIndex.from_raw(
		root_dir=args.root_dir,
		split=args.split,
		include_err=args.include_err,
		require_existing_images=True,
		limit=args.limit,
	)

	print(f"Raw root: {index.raw_root}")
	print(f"Loaded samples: {len(index)}")

	for i, sample in enumerate(index.samples[: max(0, args.preview)]):
		print(
			f"[{i}] id={sample.line_id}, writer={sample.writer_id}, "
			f"status={sample.status}, text='{sample.text[:80]}'"
		)

	if args.manifest:
		path = index.save_manifest(output_path=args.manifest, fmt=args.manifest_format)
		print(f"Manifest saved: {path}")

	if args.export_dir:
		manifest_path = index.export_subset(output_dir=args.export_dir, copy_images=args.copy_images)
		print(f"Subset exported: {manifest_path}")


if __name__ == "__main__":
	main()
