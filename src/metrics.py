from __future__ import annotations

from typing import Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    table = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        table[row][0] = row
    for col in range(cols):
        table[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if reference[row - 1] == hypothesis[col - 1] else 1
            table[row][col] = min(
                table[row - 1][col] + 1,
                table[row][col - 1] + 1,
                table[row - 1][col - 1] + cost,
            )
    return table[-1][-1]


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    total_distance = 0
    total_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        total_distance += edit_distance(list(reference), list(prediction))
        total_length += len(reference)
    if total_length == 0:
        return 0.0
    return total_distance / total_length


def word_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    total_distance = 0
    total_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        reference_words = reference.split()
        prediction_words = prediction.split()
        total_distance += edit_distance(reference_words, prediction_words)
        total_length += len(reference_words)
    if total_length == 0:
        return 0.0
    return total_distance / total_length


def cer(predictions: Sequence[str], references: Sequence[str]) -> float:
    return character_error_rate(predictions, references)


def wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    return word_error_rate(predictions, references)
