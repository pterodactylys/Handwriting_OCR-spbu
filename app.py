"""
Tkinter GUI for OCR prediction and evaluation.
Load a checkpoint, select/drag images, and see predictions instantly.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import torch
from PIL import Image, ImageTk

from data.preprocessing import ImageConfig, preprocess_image
from data.vocab import CharVocab
from models.crnn import CRNN
from models.ctc_decoder import greedy_decode


class OCRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Handwriting OCR")
        self.root.geometry("1000x700")

        self.model: Optional[CRNN] = None
        self.vocab: Optional[CharVocab] = None
        self.image_cfg: Optional[ImageConfig] = None
        self.device = torch.device("cpu")
        self.checkpoint_path: Optional[Path] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        # Top frame: checkpoint loading
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top_frame, text="Checkpoint:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.ckpt_label = tk.Label(
            top_frame, text="<none>", font=("Arial", 9), fg="gray", relief=tk.SUNKEN, width=50
        )
        self.ckpt_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        tk.Button(top_frame, text="Load Checkpoint", command=self._load_checkpoint, bg="#4CAF50", fg="white").pack(
            side=tk.LEFT, padx=5
        )

        # Middle frame: image display + prediction area
        middle_frame = tk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: image display
        left_frame = tk.Frame(middle_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left_frame, text="Image", font=("Arial", 10, "bold")).pack()
        self.image_label = tk.Label(left_frame, bg="lightgray", width=400, height=300)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        tk.Button(left_frame, text="Open Image", command=self._open_image, bg="#2196F3", fg="white", height=2).pack(
            fill=tk.X, pady=(10, 0)
        )

        # Right: prediction area
        right_frame = tk.Frame(middle_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Prediction", font=("Arial", 10, "bold")).pack()
        self.pred_text = tk.Text(right_frame, height=5, font=("Courier", 11), wrap=tk.WORD)
        self.pred_text.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        tk.Label(right_frame, text="Ground Truth (optional)", font=("Arial", 10, "bold")).pack()
        self.gt_text = tk.Text(right_frame, height=5, font=("Courier", 11), wrap=tk.WORD, bg="#f0f0f0")
        self.gt_text.pack(fill=tk.BOTH, expand=True)

        # Bottom frame: controls
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(bottom_frame, text="Predict", command=self._predict, bg="#FF9800", fg="white", height=2, width=20).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(bottom_frame, text="Clear", command=self._clear, height=2, width=20).pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(bottom_frame, text="Ready", font=("Arial", 9), fg="green")
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[Path] = None

    def _load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CRNN checkpoint",
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self._set_status("Loading checkpoint...", "orange")
            ckpt = torch.load(Path(path).expanduser().resolve(), map_location=self.device)

            self.vocab = CharVocab.load(ckpt["vocab_path"])
            self.image_cfg = ImageConfig(height=int(ckpt["image_height"]), width=int(ckpt["image_width"]))
            self.model = CRNN(num_classes=self.vocab.size).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self.checkpoint_path = Path(path)
            self.ckpt_label.config(text=str(self.checkpoint_path.name), fg="black")
            self._set_status(f"Checkpoint loaded: {self.checkpoint_path.name}", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint:\n{e}")
            self._set_status("Error loading checkpoint", "red")

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.current_image = Image.open(path).convert("L")
            self.current_image_path = Path(path)
            self._display_image()
            self._check_for_ground_truth()
            self._set_status(f"Loaded: {self.current_image_path.name}", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            self._set_status("Error loading image", "red")

    def _display_image(self) -> None:
        if self.current_image is None:
            return
        img = self.current_image.copy()
        img.thumbnail((400, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.photo = photo

    def _check_for_ground_truth(self) -> None:
        self.gt_text.config(state=tk.NORMAL)
        self.gt_text.delete("1.0", tk.END)

        if not self.current_image_path:
            return

        cwd = Path.cwd()
        for manifest_file in [
            cwd / "artifacts/data_small/train.jsonl",
            cwd / "artifacts/data_small/val.jsonl",
            cwd / "artifacts/data_small/test.jsonl",
            cwd / "artifacts/data_full/train.jsonl",
            cwd / "artifacts/data_full/val.jsonl",
            cwd / "artifacts/data_full/test.jsonl",
        ]:
            if not manifest_file.exists():
                continue
            try:
                with manifest_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        if Path(row["image_path"]).resolve() == self.current_image_path.resolve():
                            text = row.get("text", "")
                            self.gt_text.insert(tk.END, text)
                            self.gt_text.config(state=tk.DISABLED)
                            return
            except Exception:
                pass

    def _predict(self) -> None:
        if self.model is None or self.vocab is None or self.image_cfg is None:
            messagebox.showwarning("Warning", "Please load a checkpoint first.")
            return

        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        try:
            self._set_status("Predicting...", "orange")
            tensor = preprocess_image(self.current_image, self.image_cfg).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                log_probs = logits.log_softmax(dim=2)
                pred_ids = greedy_decode(log_probs, blank_idx=self.vocab.blank_idx)[0]

            pred_text = self.vocab.decode(pred_ids)
            self.pred_text.config(state=tk.NORMAL)
            self.pred_text.delete("1.0", tk.END)
            self.pred_text.insert(tk.END, pred_text if pred_text else "<empty>")
            self.pred_text.config(state=tk.DISABLED)
            self._set_status("Prediction complete", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")
            self._set_status("Prediction failed", "red")

    def _clear(self) -> None:
        self.pred_text.config(state=tk.NORMAL)
        self.pred_text.delete("1.0", tk.END)
        self.gt_text.config(state=tk.NORMAL)
        self.gt_text.delete("1.0", tk.END)
        self.image_label.config(image="")
        self.image_label.photo = None
        self.current_image = None
        self.current_image_path = None
        self._set_status("Cleared", "blue")

    def _set_status(self, msg: str, color: str = "black") -> None:
        self.status_label.config(text=msg, fg=color)
        self.root.update_idletasks()


def main() -> None:
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
