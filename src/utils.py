from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional during static checks
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable: Iterable | None = None, **_: object) -> Iterable | None:
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for training and inference. "
            "Install the packages from requirements.txt in your local environment."
        )


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_parent_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def load_config(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: dict[str, Any], path_like: str | Path) -> Path:
    path = ensure_parent_dir(path_like)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def load_json(path_like: str | Path) -> Any:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path_like: str | Path) -> Path:
    path = ensure_parent_dir(path_like)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def select_device() -> Any:
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: Any) -> int:
    if torch is None:
        return 0
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
