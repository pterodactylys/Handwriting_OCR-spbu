from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import AugmentationConfig, IAMLineDataset, ctc_collate_fn
from src.decode import greedy_decode
from src.metrics import cer, wer
from src.model import CRNN
from src.tokenizer import CharacterTokenizer
from src.utils import (
    AverageMeter,
    count_parameters,
    ensure_dir,
    load_config,
    require_torch,
    resolve_path,
    save_config,
    seed_everything,
    select_device,
    tqdm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CRNN+CTC baseline on IAM lines.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def build_datasets(
    config: dict,
    tokenizer: CharacterTokenizer,
) -> tuple[IAMLineDataset, IAMLineDataset]:
    overfit_num_samples = int(config.get("overfit_num_samples", 0))
    augmentation_payload = config.get("augmentation", {})
    use_train_augmentations = bool(config.get("train_augmentations", True))

    if overfit_num_samples > 0:
        print(
            f"Overfit mode is enabled on {overfit_num_samples} samples. "
            "Train and validation will use the same subset, and augmentations are disabled."
        )
        train_manifest = config["train_csv"]
        val_manifest = config["train_csv"]
        train_max_samples = overfit_num_samples
        val_max_samples = overfit_num_samples
        use_train_augmentations = False
    else:
        train_manifest = config["train_csv"]
        val_manifest = config["val_csv"]
        train_max_samples = 0
        val_max_samples = 0

    train_dataset = IAMLineDataset(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        img_height=int(config["img_height"]),
        train=True,
        augmentations=AugmentationConfig.from_dict(
            augmentation_payload,
            enabled=use_train_augmentations,
        ),
        max_samples=train_max_samples,
    )
    val_dataset = IAMLineDataset(
        manifest_path=val_manifest,
        tokenizer=tokenizer,
        img_height=int(config["img_height"]),
        train=False,
        augmentations=AugmentationConfig(enabled=False),
        max_samples=val_max_samples,
    )
    return train_dataset, val_dataset


def build_dataloaders(
    config: dict,
    tokenizer: CharacterTokenizer,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(config, tokenizer)
    pin_memory = torch.cuda.is_available()
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate_fn,
    )
    return train_loader, val_loader


def save_checkpoint(
    path: Path,
    *,
    model: CRNN,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_cer: float,
    config: dict,
    tokenizer: CharacterTokenizer,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_cer": best_cer,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": config,
        "model_config": {
            "hidden_size": int(config["model"]["hidden_size"]),
            "lstm_layers": int(config["model"]["lstm_layers"]),
            "dropout": float(config["model"]["dropout"]),
            "num_classes": tokenizer.num_classes,
        },
    }
    torch.save(checkpoint, path)


def save_prediction_examples(examples: list[dict[str, str]], output_path: Path) -> None:
    if not examples:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(examples).to_csv(output_path, index=False)


def run_sanity_checks(
    *,
    model: CRNN,
    train_loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> None:
    batch = next(iter(train_loader))
    images = batch["images"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    image_widths = batch["image_widths"].to(device)

    model.eval()
    with torch.no_grad():
        log_probs, output_lengths = model(images, image_widths)
        loss = criterion(log_probs, targets, output_lengths, target_lengths)
        predictions = greedy_decode(log_probs.cpu(), tokenizer, output_lengths.cpu())

    assert log_probs.shape[1] == images.shape[0], "Batch dimension mismatch in CRNN output."
    assert len(predictions) == images.shape[0], "Greedy decode returned a wrong batch size."
    assert torch.all(output_lengths > 0), "CTC input lengths must be positive."
    print(
        "Sanity check passed: "
        f"images={tuple(images.shape)}, log_probs={tuple(log_probs.shape)}, loss={loss.item():.4f}"
    )


def train_one_epoch(
    *,
    model: CRNN,
    train_loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: Adam,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    clip_grad_norm: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    progress = tqdm(train_loader, desc="Train", leave=False)

    for batch in progress:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        image_widths = batch["image_widths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            log_probs, output_lengths = model(images, image_widths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)

        scaler.scale(loss).backward()
        if clip_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=images.size(0))
        progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    return {"loss": loss_meter.average}


def validate(
    *,
    model: CRNN,
    val_loader: DataLoader,
    criterion: nn.CTCLoss,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    use_amp: bool,
    max_examples: int,
) -> dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    predictions_all: list[str] = []
    references_all: list[str] = []
    examples: list[dict[str, str]] = []

    progress = tqdm(val_loader, desc="Val", leave=False)
    with torch.no_grad():
        for batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            image_widths = batch["image_widths"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs, output_lengths = model(images, image_widths)
                loss = criterion(log_probs, targets, output_lengths, target_lengths)

            batch_predictions = greedy_decode(
                log_probs.cpu(),
                tokenizer,
                output_lengths.cpu(),
            )
            batch_references = list(batch["texts"])

            predictions_all.extend(batch_predictions)
            references_all.extend(batch_references)
            loss_meter.update(loss.item(), n=images.size(0))

            if len(examples) < max_examples:
                for image_path, reference, prediction in zip(
                    batch["image_paths"],
                    batch_references,
                    batch_predictions,
                    strict=True,
                ):
                    examples.append(
                        {
                            "image_path": image_path,
                            "ground_truth": reference,
                            "prediction": prediction,
                        }
                    )
                    if len(examples) >= max_examples:
                        break

            progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    return {
        "loss": loss_meter.average,
        "cer": cer(predictions_all, references_all),
        "wer": wer(predictions_all, references_all),
        "examples": examples,
    }


def fit(
    *,
    model: CRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharacterTokenizer,
    config: dict,
    checkpoint_dir: Path,
    prediction_log_dir: Path,
    device: torch.device,
) -> None:
    criterion = nn.CTCLoss(blank=tokenizer.blank_index, zero_infinity=True)
    optimizer = Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("scheduler_factor", 0.5)),
        patience=int(config.get("scheduler_patience", 2)),
    )
    use_amp = bool(config.get("use_amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_grad_norm = float(config.get("clip_grad_norm", 5.0))
    max_examples = int(config.get("log_examples", 8))

    run_sanity_checks(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        tokenizer=tokenizer,
        device=device,
    )

    best_cer = float("inf")
    epochs_without_improvement = 0
    total_epochs = int(config["epochs"])
    early_stopping_patience = int(config.get("early_stopping_patience", 5))

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            clip_grad_norm=clip_grad_norm,
            use_amp=use_amp,
        )
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device,
            use_amp=use_amp,
            max_examples=max_examples,
        )
        scheduler.step(val_metrics["cer"])

        improved = val_metrics["cer"] < best_cer
        if improved:
            best_cer = float(val_metrics["cer"])
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_cer=best_cer,
                config=config,
                tokenizer=tokenizer,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_cer=best_cer,
            config=config,
            tokenizer=tokenizer,
        )
        save_prediction_examples(
            val_metrics["examples"],
            prediction_log_dir / f"val_predictions_epoch_{epoch:03d}.csv",
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"CER={val_metrics['cer']:.4f} | "
            f"WER={val_metrics['wer']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping triggered because validation CER has not improved "
                f"for {early_stopping_patience} epochs."
            )
            break


def main() -> None:
    require_torch()
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    device = select_device()
    checkpoint_dir = ensure_dir(config["checkpoint_dir"])
    prediction_log_dir = ensure_dir(
        config.get("prediction_log_dir", str(checkpoint_dir / "validation_predictions"))
    )

    tokenizer = CharacterTokenizer.from_manifest(resolve_path(config["train_csv"]))
    alphabet_path = resolve_path(config.get("alphabet_path", "artifacts/alphabet.json"))
    tokenizer.save(alphabet_path)
    tokenizer.save(checkpoint_dir / "alphabet.json")
    save_config(config, checkpoint_dir / "config.yaml")

    train_loader, val_loader = build_dataloaders(config, tokenizer)
    model = CRNN(
        num_classes=tokenizer.num_classes,
        hidden_size=int(config["model"]["hidden_size"]),
        lstm_layers=int(config["model"]["lstm_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Alphabet size: {tokenizer.num_classes}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        checkpoint_dir=checkpoint_dir,
        prediction_log_dir=prediction_log_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
