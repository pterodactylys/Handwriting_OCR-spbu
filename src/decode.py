from __future__ import annotations

from typing import Sequence

import torch

from src.tokenizer import CharacterTokenizer


def collapse_ctc_tokens(token_ids: Sequence[int], blank_index: int = 0) -> list[int]:
    collapsed: list[int] = []
    previous = None
    for token_id in token_ids:
        if token_id != blank_index and token_id != previous:
            collapsed.append(int(token_id))
        previous = int(token_id)
    return collapsed


def greedy_decode(
    log_probs: torch.Tensor,
    tokenizer: CharacterTokenizer,
    input_lengths: Sequence[int] | torch.Tensor | None = None,
) -> list[str]:
    if log_probs.ndim != 3:
        raise ValueError("Expected log_probs with shape [T, B, C].")

    best_path = log_probs.argmax(dim=-1)  # [T, B]
    time_steps, batch_size = best_path.shape
    if input_lengths is None:
        input_lengths = [time_steps] * batch_size
    if isinstance(input_lengths, torch.Tensor):
        input_lengths = input_lengths.detach().cpu().tolist()

    predictions: list[str] = []
    for batch_index in range(batch_size):
        sequence = best_path[: input_lengths[batch_index], batch_index].detach().cpu().tolist()
        collapsed = collapse_ctc_tokens(sequence, blank_index=tokenizer.blank_index)
        predictions.append(tokenizer.ids_to_text(collapsed, remove_blank=True))
    return predictions
