from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from src.utils import load_json, save_json

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional during static checks
    torch = None


class CharacterTokenizer:
    def __init__(self, symbols: Sequence[str], blank_token: str = "<BLANK>") -> None:
        unique_symbols = sorted(set(symbols))
        if blank_token in unique_symbols:
            raise ValueError("blank_token must not be part of the regular alphabet.")

        self.blank_token = blank_token
        self.blank_index = 0
        self.symbols = unique_symbols
        self.idx_to_char = {self.blank_index: self.blank_token}
        for offset, symbol in enumerate(self.symbols, start=1):
            self.idx_to_char[offset] = symbol
        self.char_to_idx = {symbol: index for index, symbol in self.idx_to_char.items()}

    @classmethod
    def from_texts(cls, texts: Iterable[str], blank_token: str = "<BLANK>") -> "CharacterTokenizer":
        symbols = sorted({character for text in texts for character in text})
        return cls(symbols=symbols, blank_token=blank_token)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, blank_token: str = "<BLANK>") -> "CharacterTokenizer":
        texts: list[str] = []
        with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                texts.append(row["text"])
        return cls.from_texts(texts=texts, blank_token=blank_token)

    @classmethod
    def load(cls, path_like: str | Path) -> "CharacterTokenizer":
        payload = load_json(path_like)
        tokenizer = cls(symbols=payload["symbols"], blank_token=payload["blank_token"])
        if tokenizer.blank_index != payload["blank_index"]:
            raise ValueError("Loaded alphabet has an unexpected blank index.")
        return tokenizer

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_char)

    def save(self, path_like: str | Path) -> Path:
        return save_json(
            {
                "blank_token": self.blank_token,
                "blank_index": self.blank_index,
                "symbols": self.symbols,
            },
            path_like,
        )

    def encode(self, text: str) -> list[int]:
        unknown = sorted({character for character in text if character not in self.char_to_idx})
        if unknown:
            raise ValueError(f"Text contains characters outside the alphabet: {unknown}")
        return [self.char_to_idx[character] for character in text]

    def text_to_tensor(self, text: str) -> "torch.Tensor":
        if torch is None:
            raise ImportError("PyTorch is required for text_to_tensor().")
        return torch.tensor(self.encode(text), dtype=torch.long)

    def ids_to_text(
        self,
        ids: Sequence[int],
        *,
        remove_blank: bool = True,
    ) -> str:
        characters: list[str] = []
        for token_id in ids:
            if remove_blank and token_id == self.blank_index:
                continue
            if token_id not in self.idx_to_char:
                raise ValueError(f"Unknown token id: {token_id}")
            if token_id == self.blank_index:
                continue
            characters.append(self.idx_to_char[token_id])
        return "".join(characters)

    def decode_greedy(self, ids: Sequence[int]) -> str:
        collapsed: list[int] = []
        previous = None
        for token_id in ids:
            if token_id != self.blank_index and token_id != previous:
                collapsed.append(token_id)
            previous = token_id
        return self.ids_to_text(collapsed, remove_blank=True)
