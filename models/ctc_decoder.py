from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def greedy_decode(log_probs: torch.Tensor, blank_idx: int) -> List[List[int]]:
    """CTC greedy decoding.

    Args:
        log_probs: Tensor [T, B, C] after log_softmax.
        blank_idx: Index of CTC blank token.
    """
    best = log_probs.argmax(dim=2)  # [T, B]
    results: List[List[int]] = []

    for b in range(best.size(1)):
        seq = best[:, b].tolist()
        decoded = []
        prev = None
        for idx in seq:
            if idx != blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx
        results.append(decoded)

    return results
