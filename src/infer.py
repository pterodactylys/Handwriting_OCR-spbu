from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import preprocess_line_image
from src.decode import greedy_decode
from src.model import CRNN
from src.tokenizer import CharacterTokenizer
from src.utils import require_torch, resolve_path, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRNN+CTC inference on a single line image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--alphabet", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, tokenizer: CharacterTokenizer, device: torch.device) -> tuple[CRNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = CRNN(
        num_classes=tokenizer.num_classes,
        hidden_size=int(model_config["hidden_size"]),
        lstm_layers=int(model_config["lstm_layers"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def main() -> None:
    require_torch()
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    device = select_device()

    alphabet_path = (
        resolve_path(args.alphabet)
        if args.alphabet is not None
        else checkpoint_path.parent / "alphabet.json"
    )
    tokenizer = CharacterTokenizer.load(alphabet_path)
    model, checkpoint = load_model(checkpoint_path, tokenizer, device)

    config = checkpoint.get("config", {})
    img_height = int(config.get("img_height", 32))
    image_tensor, original_size, resized_size = preprocess_line_image(
        args.image,
        img_height=img_height,
        apply_augmentations=False,
    )
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_widths = torch.tensor([resized_size[0]], dtype=torch.long, device=device)

    with torch.no_grad():
        log_probs, output_lengths = model(image_tensor, image_widths)
        prediction = greedy_decode(log_probs.cpu(), tokenizer, output_lengths.cpu())[0]

    print(prediction)
    if args.debug:
        print(f"image_path={resolve_path(args.image)}")
        print(f"original_size={original_size}")
        print(f"resized_size={resized_size}")
        print(f"ctc_sequence_length={int(output_lengths.item())}")


if __name__ == "__main__":
    main()
