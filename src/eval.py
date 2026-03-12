from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import AugmentationConfig, IAMLineDataset, ctc_collate_fn
from src.decode import greedy_decode
from src.metrics import cer, wer
from src.model import CRNN
from src.tokenizer import CharacterTokenizer
from src.utils import ensure_parent_dir, load_config, require_torch, resolve_path, select_device, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CRNN+CTC checkpoint on IAM lines.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--alphabet", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--test-csv", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="artifacts/test_predictions.csv")
    return parser.parse_args()


def resolve_alphabet_path(args: argparse.Namespace, checkpoint_path: Path, config: dict) -> Path:
    if args.alphabet is not None:
        return resolve_path(args.alphabet)
    candidate = checkpoint_path.parent / "alphabet.json"
    if candidate.exists():
        return candidate
    return resolve_path(config.get("alphabet_path", "artifacts/alphabet.json"))


def build_model_from_checkpoint(checkpoint: dict, tokenizer: CharacterTokenizer, device: torch.device) -> CRNN:
    model_config = checkpoint["model_config"]
    model = CRNN(
        num_classes=tokenizer.num_classes,
        hidden_size=int(model_config["hidden_size"]),
        lstm_layers=int(model_config["lstm_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def evaluate(
    *,
    model: CRNN,
    loader: DataLoader,
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> tuple[float, float, float, list[dict[str, str]]]:
    criterion = nn.CTCLoss(blank=tokenizer.blank_index, zero_infinity=True)
    loss_total = 0.0
    sample_count = 0
    predictions_all: list[str] = []
    references_all: list[str] = []
    rows: list[dict[str, str]] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            image_widths = batch["image_widths"].to(device)

            log_probs, output_lengths = model(images, image_widths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            predictions = greedy_decode(log_probs.cpu(), tokenizer, output_lengths.cpu())

            loss_total += loss.item() * images.size(0)
            sample_count += images.size(0)
            predictions_all.extend(predictions)
            references_all.extend(batch["texts"])

            for image_path, ground_truth, prediction in zip(
                batch["image_paths"],
                batch["texts"],
                predictions,
                strict=True,
            ):
                rows.append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                    }
                )

    avg_loss = loss_total / max(sample_count, 1)
    return avg_loss, cer(predictions_all, references_all), wer(predictions_all, references_all), rows


def main() -> None:
    require_torch()
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = checkpoint.get("config", {})
        config_path = checkpoint_path.parent / "config.yaml"
        if not config and config_path.exists():
            config = load_config(config_path)

    alphabet_path = resolve_alphabet_path(args, checkpoint_path, config)
    tokenizer = CharacterTokenizer.load(alphabet_path)

    test_csv = resolve_path(args.test_csv or config["test_csv"])
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 16))
    num_workers = int(
        args.num_workers if args.num_workers is not None else config.get("num_workers", 0)
    )
    img_height = int(config.get("img_height", 32))

    dataset = IAMLineDataset(
        manifest_path=test_csv,
        tokenizer=tokenizer,
        img_height=img_height,
        train=False,
        augmentations=AugmentationConfig(enabled=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ctc_collate_fn,
    )

    model = build_model_from_checkpoint(checkpoint, tokenizer, device)
    avg_loss, test_cer, test_wer, rows = evaluate(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        device=device,
    )

    output_csv = ensure_parent_dir(args.output_csv)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Alphabet:   {alphabet_path}")
    print(f"Test CSV:   {test_csv}")
    print(f"Loss:       {avg_loss:.4f}")
    print(f"CER:        {test_cer:.4f}")
    print(f"WER:        {test_wer:.4f}")
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()
