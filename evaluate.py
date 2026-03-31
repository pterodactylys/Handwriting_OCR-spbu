from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import ImageConfig, preprocess_image
from data.vocab import CharVocab
from models.crnn import CRNN
from models.ctc_decoder import greedy_decode
from utils.metrics import cer, wer


@dataclass
class ManifestSample:
    image_path: str
    text: str


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_cfg: ImageConfig):
        self.samples: List[ManifestSample] = []
        self.image_cfg = image_cfg

        path = Path(manifest_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                row = json.loads(line)
                self.samples.append(ManifestSample(image_path=row["image_path"], text=row["text"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("L")
        return preprocess_image(image, self.image_cfg), sample.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CRNN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="artifacts/data/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(Path(args.checkpoint).expanduser().resolve(), map_location=device)
    vocab = CharVocab.load(ckpt["vocab_path"])
    image_cfg = ImageConfig(height=int(ckpt["image_height"]), width=int(ckpt["image_width"]))

    model = CRNN(num_classes=vocab.size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = ManifestDataset(args.manifest, image_cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    cer_scores = []
    wer_scores = []

    with torch.no_grad():
        for images, texts in loader:
            images = images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            decoded = greedy_decode(log_probs, blank_idx=vocab.blank_idx)
            preds = [vocab.decode(ids) for ids in decoded]

            for ref, hyp in zip(texts, preds):
                cer_scores.append(cer(ref, hyp))
                wer_scores.append(wer(ref, hyp))

    mean_cer = sum(cer_scores) / max(1, len(cer_scores))
    mean_wer = sum(wer_scores) / max(1, len(wer_scores))

    print(f"Samples: {len(ds)}")
    print(f"CER: {mean_cer:.4f}")
    print(f"WER: {mean_wer:.4f}")


if __name__ == "__main__":
    main()
