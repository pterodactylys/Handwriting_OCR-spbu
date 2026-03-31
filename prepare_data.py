from __future__ import annotations

import argparse
from pathlib import Path

from data.dataset import IAMLinesIndex
from data.vocab import CharVocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare IAM manifests and vocabulary")
    parser.add_argument("--root_dir", type=str, default="data", help="Path to project root or raw/")
    parser.add_argument("--output_dir", type=str, default="artifacts/data")
    parser.add_argument("--include_err", action="store_true")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": args.train_limit,
        "val": args.val_limit,
        "test": args.test_limit,
    }

    indices = {}
    for split, limit in splits.items():
        index = IAMLinesIndex.from_raw(
            root_dir=args.root_dir,
            split=split,
            include_err=args.include_err,
            require_existing_images=True,
            limit=limit,
        )
        indices[split] = index
        manifest = output_dir / f"{split}.jsonl"
        index.save_manifest(manifest, fmt="jsonl")
        print(f"Saved {split}: {manifest} ({len(index)} samples)")

    train_texts = [sample.text for sample in indices["train"].samples]
    vocab = CharVocab.build(train_texts)
    vocab_path = vocab.save(output_dir / "vocab.json")
    print(f"Saved vocab: {vocab_path} ({vocab.size} symbols)")


if __name__ == "__main__":
    main()
