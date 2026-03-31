from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CharVocab:
    stoi: dict
    itos: List[str]
    blank_token: str = "<blank>"

    @property
    def blank_idx(self) -> int:
        return self.stoi[self.blank_token]

    @property
    def size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharVocab":
        alphabet = sorted(set("".join(texts)))
        itos = ["<blank>"] + alphabet
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            raise ValueError(f"Text contains unseen symbols: {sorted(set(unknown))}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            token = self.itos[idx]
            if token != self.blank_token:
                chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos, "blank_token": self.blank_token}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        content = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        itos = content["itos"]
        blank_token = content.get("blank_token", "<blank>")
        stoi = {ch: i for i, ch in enumerate(itos)}
        if blank_token not in stoi:
            raise ValueError("blank_token is missing from loaded vocabulary")
        return cls(stoi=stoi, itos=itos, blank_token=blank_token)
