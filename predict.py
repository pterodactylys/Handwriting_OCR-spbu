from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from data.preprocessing import ImageConfig, preprocess_image
from data.vocab import CharVocab
from models.crnn import CRNN
from models.ctc_decoder import greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR inference on one line image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
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

    image = Image.open(Path(args.image).expanduser().resolve()).convert("L")
    tensor = preprocess_image(image, image_cfg).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        log_probs = logits.log_softmax(dim=2)
        pred_ids = greedy_decode(log_probs, blank_idx=vocab.blank_idx)[0]

    pred_text = vocab.decode(pred_ids)
    print(pred_text if pred_text else "<empty prediction>")


if __name__ == "__main__":
    main()
