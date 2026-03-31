from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.preprocessing import ImageConfig, preprocess_image
from data.vocab import CharVocab
from models.crnn import CRNN
from models.ctc_decoder import greedy_decode
from utils.metrics import cer


@dataclass
class ManifestSample:
    image_path: str
    text: str


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_cfg: ImageConfig):
        self.image_cfg = image_cfg
        self.samples: List[ManifestSample] = []

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
        tensor = preprocess_image(image, self.image_cfg)
        return tensor, sample.text


def collate_batch(batch: List[Tuple[torch.Tensor, str]], vocab: CharVocab):
    images = torch.stack([x[0] for x in batch], dim=0)
    texts = [x[1] for x in batch]

    targets = []
    target_lengths = []
    for text in texts:
        encoded = vocab.encode(text)
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    targets_tensor = torch.tensor(targets, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    return images, texts, targets_tensor, target_lengths_tensor


def evaluate_subset(model: nn.Module, loader: DataLoader, vocab: CharVocab, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    cer_values = []

    with torch.no_grad():
        for i, (images, texts, _, _) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            decoded_ids = greedy_decode(log_probs, blank_idx=vocab.blank_idx)
            preds = [vocab.decode(seq) for seq in decoded_ids]
            cer_values.extend(cer(ref, hyp) for ref, hyp in zip(texts, preds))

    return float(sum(cer_values) / max(1, len(cer_values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN baseline with CTC")
    parser.add_argument("--train_manifest", type=str, default="artifacts/data/train.jsonl")
    parser.add_argument("--val_manifest", type=str, default="artifacts/data/val.jsonl")
    parser.add_argument("--vocab", type=str, default="artifacts/data/vocab.json")
    parser.add_argument("--output", type=str, default="artifacts/checkpoints/crnn_baseline.pt")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    vocab = CharVocab.load(args.vocab)
    image_cfg = ImageConfig(height=args.height, width=args.width)

    train_ds = ManifestDataset(args.train_manifest, image_cfg)
    val_ds = ManifestDataset(args.val_manifest, image_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab),
    )

    model = CRNN(num_classes=vocab.size).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_cer = float("inf")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for images, _, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=2)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = sum(losses) / max(1, len(losses))
        val_cer = evaluate_subset(model, val_loader, vocab, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_cer={val_cer:.4f}")

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_path": str(Path(args.vocab).expanduser().resolve()),
                    "image_height": args.height,
                    "image_width": args.width,
                    "best_val_cer": best_val_cer,
                },
                output_path,
            )
            print(f"Saved checkpoint: {output_path}")


if __name__ == "__main__":
    main()
