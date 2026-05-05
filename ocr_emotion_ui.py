from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tkinter as tk
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageTk


ROOT = Path(__file__).resolve().parent
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_EXAMPLES_DIR = ROOT / "data" / "inference_examples"

CONTENT_CROP_THRESHOLD = 245
CONTENT_CROP_PADDING = 6
MASK_THRESHOLD = 185
MIN_COMPONENT_AREA = 10
THRESHOLD_UPSCALE_FACTOR = 2
THRESHOLD_MEDIAN_SIZE = 3
WORD_GAP_MIN_WIDTH = 6
MAX_WORDS_PER_CHUNK = 2
WORD_CHUNK_PADDING = 4
PREVIEW_MAX_SIZE = (420, 280)
GRAPH_SIZE = (1200, 280)
GRID_PREVIEW_SIZE = (320, 220)

LEXICON_PATH = ROOT / "data" / "lexicons" / "RusEmoLex.csv"
WORDFREQ_FREQ_PATH = ROOT / "data" / "lexicons" / "wordfreq_ru.tsv"

EMOTION_ORDER = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]
EMOTION_COLORS = {
    "joy": "#f4b400",
    "sadness": "#4285f4",
    "fear": "#7e57c2",
    "anger": "#db4437",
    "surprise": "#0f9d58",
    "disgust": "#8d6e63",
}
EMOTION_UI_LABELS = {
    "joy": "радость",
    "sadness": "грусть",
    "fear": "страх",
    "anger": "злость",
    "surprise": "удивление",
    "disgust": "отвращение",
    "neutral": "нейтрально",
    "unassigned": "не определено",
}
VARIANT_ORDER = [
    ("baseline", "Базовая"),
    ("baseline_segmented", "Базовая + чанки"),
    ("thresholded_segmented", "Бинаризация + чанки"),
]


try:
    from wordfreq import iter_wordlist, zipf_frequency

    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

try:
    import pymorphy3  # noqa: F401

    PYMORPHY3_AVAILABLE = True
except ImportError:
    PYMORPHY3_AVAILABLE = False


class RussianArgumentParser(argparse.ArgumentParser):
    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "использование:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "использование:", 1)


class CharacterMapper:
    def __init__(self, samples):
        unique_chars = set()
        for sample in samples:
            unique_chars.update(sample["transcription"])
        self.chars = sorted(unique_chars)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = ""
        self.num_classes = len(self.chars) + 1

    def encode(self, text):
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
        return indices

    def decode(self, indices):
        chars = []
        prev_idx = None
        for idx in indices:
            if idx != 0 and idx != prev_idx and idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            prev_idx = idx
        return "".join(chars)


class CRNN(nn.Module):
    """CNN-BiLSTM-CTC with the same off-by-one output size as in the notebooks."""

    def __init__(self, img_height=64, num_chars=80, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.map_to_seq = nn.LazyLinear(hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

    def forward(self, x):
        conv = self.cnn(x)
        batch_size, channels, height, width = conv.size()
        conv = conv.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)
        seq = self.map_to_seq(conv)
        rnn_out, _ = self.rnn(seq)
        output = self.fc(rnn_out)
        return torch.nn.functional.log_softmax(output, dim=2)


def decode_predictions(outputs, char_mapper):
    _, max_indices = torch.max(outputs, dim=2)
    predictions = []
    for seq in max_indices:
        predictions.append(char_mapper.decode(seq.cpu().numpy()))
    return predictions


def enhance_handwriting_image(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    img = ImageEnhance.Brightness(img).enhance(1.12)
    return img


def build_binary_mask(img: Image.Image, threshold: int = CONTENT_CROP_THRESHOLD) -> np.ndarray:
    img_arr = np.array(img, dtype=np.uint8)
    return np.where(img_arr > threshold, 255, 0).astype(np.uint8)


def compute_content_bbox_from_binary(
    binary_arr: np.ndarray,
    padding: int = CONTENT_CROP_PADDING,
    background: int = 255,
) -> tuple[int, int, int, int]:
    foreground_coords = np.argwhere(binary_arr != background)
    if foreground_coords.size == 0:
        height, width = binary_arr.shape
        return (0, 0, width, height)

    y_min, x_min = foreground_coords.min(axis=0)
    y_max, x_max = foreground_coords.max(axis=0)
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(binary_arr.shape[1], int(x_max) + padding + 1)
    y_max = min(binary_arr.shape[0], int(y_max) + padding + 1)
    return (x_min, y_min, x_max, y_max)


def crop_to_content(
    img: Image.Image,
    threshold: int = CONTENT_CROP_THRESHOLD,
    padding: int = CONTENT_CROP_PADDING,
) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    binary_arr = build_binary_mask(img, threshold=threshold)
    bbox = compute_content_bbox_from_binary(binary_arr, padding=padding)
    return img.crop(bbox), binary_arr[bbox[1] : bbox[3], bbox[0] : bbox[2]], bbox


def build_tensor_from_pil(
    img: Image.Image,
    img_height: int = 64,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> tuple[torch.Tensor, Image.Image, np.ndarray]:
    width, height = img.size
    new_width = max(4, int(round(img_height * width / max(height, 1))))
    img_resized = img.resize((new_width, img_height), resample)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr_norm = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0)
    return tensor, img_resized, arr


def remove_small_black_components(binary_arr: np.ndarray, min_area: int = 0) -> np.ndarray:
    if min_area <= 0:
        return binary_arr

    foreground = binary_arr == 0
    visited = np.zeros_like(foreground, dtype=bool)
    height, width = foreground.shape
    cleaned = binary_arr.copy()

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not foreground[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not foreground[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) < min_area:
                for cy, cx in component:
                    cleaned[cy, cx] = 255
    return cleaned


def crop_binary_pil_to_content(img: Image.Image, padding: int = 0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    binary_arr = np.array(img, dtype=np.uint8)
    bbox = compute_content_bbox_from_binary(binary_arr, padding=padding)
    return img.crop(bbox), bbox


def find_word_bounds(binary_arr: np.ndarray, min_gap_width: int = WORD_GAP_MIN_WIDTH) -> list[tuple[int, int]]:
    has_foreground = np.any(binary_arr == 0, axis=0)
    content_cols = np.flatnonzero(has_foreground)
    if content_cols.size == 0:
        return [(0, binary_arr.shape[1])]

    left = int(content_cols[0])
    right = int(content_cols[-1])
    effective_min_gap = max(min_gap_width, int(round(binary_arr.shape[0] * 0.01)))
    min_word_width = max(4, int(round(binary_arr.shape[0] * 0.10)))
    bounds = []
    segment_start = left
    x = left

    while x <= right:
        if has_foreground[x]:
            x += 1
            continue
        gap_start = x
        while x <= right and not has_foreground[x]:
            x += 1
        gap_end = x - 1
        if gap_end - gap_start + 1 >= effective_min_gap:
            split_x = (gap_start + gap_end) // 2
            if split_x - segment_start >= min_word_width:
                bounds.append((segment_start, split_x))
            segment_start = split_x + 1

    if right + 1 - segment_start >= min_word_width:
        bounds.append((segment_start, right + 1))
    return bounds or [(left, right + 1)]


def group_word_bounds(
    word_bounds: list[tuple[int, int]],
    max_words_per_chunk: int = MAX_WORDS_PER_CHUNK,
    padding: int = WORD_CHUNK_PADDING,
    max_width: int | None = None,
) -> list[tuple[int, int]]:
    if not word_bounds:
        return []
    if max_width is None:
        max_width = word_bounds[-1][1]

    chunk_bounds = []
    for index in range(0, len(word_bounds), max_words_per_chunk):
        chunk_words = word_bounds[index : index + max_words_per_chunk]
        start = max(0, chunk_words[0][0] - padding)
        end = min(max_width, chunk_words[-1][1] + padding)
        chunk_bounds.append((start, end))
    return chunk_bounds


def stitch_chunk_previews(chunk_images: list[Image.Image], gap: int = 12, background: int = 255) -> Image.Image:
    if not chunk_images:
        return Image.new("L", (64, 64), color=background)

    max_height = max(img.height for img in chunk_images)
    total_width = sum(img.width for img in chunk_images) + gap * max(0, len(chunk_images) - 1)
    canvas = Image.new("L", (total_width, max_height), color=background)

    x_offset = 0
    for chunk_img in chunk_images:
        y_offset = (max_height - chunk_img.height) // 2
        canvas.paste(chunk_img, (x_offset, y_offset))
        x_offset += chunk_img.width + gap
    return canvas


def draw_word_bound_preview(thresholded_img: Image.Image, word_bounds: list[tuple[int, int]]) -> Image.Image:
    preview = thresholded_img.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for start, end in word_bounds:
        draw.line([(start, 0), (start, preview.height)], fill=(0, 191, 255), width=2)
        x = max(start, end - 1)
        draw.line([(x, 0), (x, preview.height)], fill=(0, 191, 255), width=2)
    return preview


def segment_words_from_source(
    source_img: Image.Image,
    binary_arr: np.ndarray,
    *,
    min_gap_width: int = WORD_GAP_MIN_WIDTH,
    max_words_per_chunk: int = MAX_WORDS_PER_CHUNK,
) -> dict[str, object]:
    word_bounds = find_word_bounds(binary_arr, min_gap_width=min_gap_width)
    chunk_bounds = group_word_bounds(
        word_bounds,
        max_words_per_chunk=max_words_per_chunk,
        padding=WORD_CHUNK_PADDING,
        max_width=source_img.width,
    )

    chunk_images = []
    for start, end in chunk_bounds:
        raw_chunk_img = source_img.crop((start, 0, end, source_img.height))
        raw_chunk_binary = binary_arr[:, start:end]
        chunk_bbox = compute_content_bbox_from_binary(raw_chunk_binary, padding=max(1, WORD_CHUNK_PADDING // 2))
        chunk_images.append(raw_chunk_img.crop(chunk_bbox))

    return {
        "word_bounds": word_bounds,
        "chunk_bounds": chunk_bounds,
        "chunk_images": chunk_images,
    }


def draw_segmented_preview(source_img: Image.Image, chunk_bounds: list[tuple[int, int]]) -> Image.Image:
    preview = source_img.convert("RGB")
    draw = ImageDraw.Draw(preview)
    for start, end in chunk_bounds:
        draw.line((start, 0, start, preview.height), fill=(0, 191, 255), width=2)
        draw.line((end, 0, end, preview.height), fill=(0, 191, 255), width=2)
    return preview


@dataclass
class CorrectionCandidate:
    word: str
    score: float
    frequency: int
    edit_distance: int
    jaccard: float


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    term: str
    emotion: str
    part_of_speech: str = ""
    source_category: str = ""
    source_count: int = 1
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchExplanation:
    original_token: str
    normalized_token: str
    corrected_token: str
    lemma: str
    matched_term: str
    emotion: str
    weight: float
    negated: bool
    intensified: bool
    source_category: str
    source_count: int


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    original_text: str
    normalized_text: str
    corrected_text: str
    content_token_count: int
    enough_text: bool
    emotion_scores: dict[str, float]
    polarity_score: float
    dominant_emotion: str | None
    matched: tuple[MatchExplanation, ...] = ()


@dataclass(slots=True)
class AnalyzerConfig:
    min_content_tokens: int = 5
    correction_min_length: int = 4
    negation_window: int = 3
    intensifier_window: int = 1
    contrast_boost: float = 1.25
    pre_contrast_discount: float = 0.8
    intensity_multiplier: float = 1.5
    downtone_multiplier: float = 0.65
    negation_multiplier: float = -0.75
    collapse_yo: bool = True
    negations: frozenset[str] = field(
        default_factory=lambda: frozenset({"не", "нет", "ни", "никогда", "ничуть", "нисколько", "совсем_не", "вовсе_не"})
    )
    intensifiers: frozenset[str] = field(
        default_factory=lambda: frozenset({"очень", "крайне", "сильно", "ужасно", "совсем", "чрезвычайно"})
    )
    downtoners: frozenset[str] = field(
        default_factory=lambda: frozenset({"немного", "слегка", "чуть", "едва", "отчасти"})
    )
    contrast_markers: frozenset[str] = field(default_factory=lambda: frozenset({"но", "однако", "зато"}))


EMOTION_LABEL_MAP = {
    "joy": "joy",
    "радость": "joy",
    "happiness": "joy",
    "sadness": "sadness",
    "грусть": "sadness",
    "печаль": "sadness",
    "fear": "fear",
    "страх": "fear",
    "anger": "anger",
    "злость": "anger",
    "гнев": "anger",
    "surprise": "surprise",
    "удивление": "surprise",
    "disgust": "disgust",
    "отвращение": "disgust",
    "neutral": "neutral",
    "нейтральный": "neutral",
    "none": "unassigned",
    "unassigned": "unassigned",
}

HEADER_ALIASES = {
    "word": "term",
    "слово": "term",
    "lemma": "term",
    "лемма": "term",
    "pos": "part_of_speech",
    "part of speech": "part_of_speech",
    "part_of_speech": "part_of_speech",
    "часть речи": "part_of_speech",
    "класс": "emotion",
    "class": "emotion",
    "emotion": "emotion",
    "emotion class": "emotion",
    "эмоция": "emotion",
    "source category": "source_category",
    "source_category": "source_category",
    "категория источников": "source_category",
    "source count": "source_count",
    "source_count": "source_count",
    "occurrence count": "source_count",
    "количество вхождений": "source_count",
    "frequency of occurrence across sources": "source_count",
    "sources": "sources",
    "источники": "sources",
    "source names": "sources",
}

POSITIONAL_COLUMNS = ("term", "part_of_speech", "emotion", "source_category", "source_count", "sources")
WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)
TOKEN_RE = WORD_RE
POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"sadness", "fear", "anger", "disgust"}


def normalize_text(value: str, collapse_yo: bool = False) -> str:
    text = " ".join(value.strip().lower().split())
    return text.replace("ё", "е") if collapse_yo else text


def format_emotion_label(emotion: str | None) -> str:
    if not emotion:
        return "нет"
    return EMOTION_UI_LABELS.get(emotion, emotion)


def normalize_word(word: str, collapse_yo: bool = True) -> str:
    normalized = word.strip().lower()
    return normalized.replace("ё", "е") if collapse_yo else normalized


def _normalize_header(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _guess_delimiter(first_line: str) -> str:
    candidates = (";", "\t", ",")
    counts = {delimiter: first_line.count(delimiter) for delimiter in candidates}
    return max(counts, key=counts.get) if any(counts.values()) else ";"


def word_trigrams(word: str) -> set[str]:
    padded = f"^{word}$"
    if len(padded) < 3:
        return {padded}
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(previous[col_index] + 1, current[col_index - 1] + 1, previous[col_index - 1] + substitution_cost)
            )
        previous = current
    return previous[-1]


def save_word_frequencies(counter: Counter[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for word, frequency in counter.most_common():
            handle.write(f"{word}\t{frequency}\n")


def build_word_frequencies_from_wordfreq(
    *,
    lang: str = "ru",
    wordlist: str = "large",
    max_words: int = 200_000,
    min_zipf: float = 2.5,
    min_word_len: int = 3,
) -> Counter[str]:
    if not WORDFREQ_AVAILABLE:
        raise RuntimeError("Пакет wordfreq не установлен")

    counter: Counter[str] = Counter()
    added = 0
    for word in iter_wordlist(lang, wordlist=wordlist):
        normalized = normalize_word(word)
        if len(normalized) < min_word_len or not WORD_RE.fullmatch(normalized):
            continue
        zipf = zipf_frequency(normalized, lang, wordlist=wordlist)
        if zipf < min_zipf:
            continue
        counter[normalized] = max(1, int(round(10**zipf)))
        added += 1
        if added >= max_words:
            break
    return counter


class TrigramSpellCorrector:
    def __init__(self, word_frequencies: Counter[str]) -> None:
        self.word_frequencies = Counter(
            {normalize_word(word): freq for word, freq in word_frequencies.items() if freq > 0}
        )
        self.trigram_index: dict[str, set[str]] = defaultdict(set)
        self.word_to_trigrams: dict[str, set[str]] = {}
        self.max_frequency = max(self.word_frequencies.values(), default=1)
        for word in self.word_frequencies:
            trigrams = word_trigrams(word)
            self.word_to_trigrams[word] = trigrams
            for trigram in trigrams:
                self.trigram_index[trigram].add(word)

    @classmethod
    def from_frequency_file(cls, path: str | Path) -> "TrigramSpellCorrector":
        counter: Counter[str] = Counter()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    counter[normalize_word(parts[0])] = int(parts[1])
                except ValueError:
                    continue
        return cls(counter)

    def suggest(
        self,
        token: str,
        *,
        top_k: int = 5,
        max_edit_distance: int = 2,
        min_jaccard: float = 0.2,
    ) -> list[CorrectionCandidate]:
        normalized = normalize_word(token)
        if not normalized:
            return []
        if normalized in self.word_frequencies:
            return [CorrectionCandidate(normalized, 1.0, self.word_frequencies[normalized], 0, 1.0)]

        token_trigrams = word_trigrams(normalized)
        candidate_words: set[str] = set()
        for trigram in token_trigrams:
            candidate_words.update(self.trigram_index.get(trigram, set()))

        scored_candidates: list[CorrectionCandidate] = []
        for candidate in candidate_words:
            candidate_trigrams = self.word_to_trigrams[candidate]
            intersection = len(token_trigrams & candidate_trigrams)
            union = len(token_trigrams | candidate_trigrams)
            jaccard = intersection / union if union else 0.0
            if jaccard < min_jaccard:
                continue
            distance = levenshtein_distance(normalized, candidate)
            if distance > max_edit_distance:
                continue
            frequency_bonus = math.log1p(self.word_frequencies[candidate]) / math.log1p(self.max_frequency)
            score = (jaccard * 0.7) + (frequency_bonus * 0.2) + ((max_edit_distance - distance + 1) * 0.1)
            scored_candidates.append(
                CorrectionCandidate(candidate, score, self.word_frequencies[candidate], distance, jaccard)
            )

        scored_candidates.sort(key=lambda item: (-item.score, item.edit_distance, -item.frequency, item.word))
        return scored_candidates[:top_k]

    def correct_token(self, token: str, *, min_score: float = 0.45, top_k: int = 5) -> str:
        suggestions = self.suggest(token, top_k=top_k)
        if not suggestions:
            return token
        return suggestions[0].word if suggestions[0].score >= min_score else token


class EmotionLexicon:
    def __init__(self, entries, collapse_yo: bool = True) -> None:
        self.collapse_yo = collapse_yo
        self.entries_by_term: dict[str, list[LexiconEntry]] = {}
        for entry in entries:
            key = normalize_text(entry.term, collapse_yo=collapse_yo)
            self.entries_by_term.setdefault(key, []).append(entry)

    def lookup(self, term: str) -> list[LexiconEntry]:
        key = normalize_text(term, collapse_yo=self.collapse_yo)
        return self.entries_by_term.get(key, [])

    @property
    def max_phrase_len(self) -> int:
        if not self.entries_by_term:
            return 1
        return max(len(term.split()) for term in self.entries_by_term)


def _read_rows(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    delimiter = _guess_delimiter(lines[0])
    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return []

    raw_headers = rows[0]
    normalized_headers = [_normalize_header(header) for header in raw_headers]
    canonical_headers = [HEADER_ALIASES.get(header, "") for header in normalized_headers]
    has_header = canonical_headers.count("term") == 1 and canonical_headers.count("emotion") == 1
    data_rows = rows[1:] if has_header else rows
    if not has_header:
        canonical_headers = list(POSITIONAL_COLUMNS[: len(raw_headers)])

    normalized_rows: list[dict[str, str]] = []
    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue
        padded_row = row + [""] * (len(canonical_headers) - len(row))
        normalized_rows.append(
            {
                column_name: padded_row[index].strip()
                for index, column_name in enumerate(canonical_headers)
                if column_name
            }
        )
    return normalized_rows


def load_rusemolex(path: str | Path, collapse_yo: bool = True) -> EmotionLexicon:
    rows = _read_rows(Path(path))
    entries = []
    for row in rows:
        term = row.get("term", "").strip()
        if not term:
            continue
        emotion_key = normalize_text(row.get("emotion", ""), collapse_yo=False)
        emotion = EMOTION_LABEL_MAP.get(emotion_key, emotion_key or "unassigned")
        try:
            source_count = int(row.get("source_count", "").strip())
        except ValueError:
            source_count = 1
        sources_raw = row.get("sources", "").strip()
        if "|" in sources_raw:
            sources = tuple(part.strip() for part in sources_raw.split("|") if part.strip())
        elif ";" in sources_raw:
            sources = tuple(part.strip() for part in sources_raw.split(";") if part.strip())
        else:
            sources = tuple(part.strip() for part in sources_raw.split(",") if part.strip())
        entries.append(
            LexiconEntry(
                term=normalize_text(term, collapse_yo=collapse_yo),
                emotion=emotion,
                part_of_speech=row.get("part_of_speech", "").strip(),
                source_category=row.get("source_category", "").strip(),
                source_count=max(1, source_count),
                sources=sources,
            )
        )
    return EmotionLexicon(entries, collapse_yo=collapse_yo)


class Pymorphy3Lemmatizer:
    def __init__(self, collapse_yo: bool = True) -> None:
        from pymorphy3 import MorphAnalyzer

        self._morph = MorphAnalyzer()
        self._collapse_yo = collapse_yo

    def lemmatize_candidates(self, token: str) -> list[str]:
        normalized = normalize_text(token, collapse_yo=self._collapse_yo)
        if not normalized:
            return [normalized]
        parsed = self._morph.parse(normalized)
        candidates = [normalized]
        for parse in parsed:
            lemma = parse.normal_form
            if self._collapse_yo:
                lemma = lemma.replace("ё", "е")
            if lemma not in candidates:
                candidates.append(lemma)
        return candidates

    def lemmatize(self, token: str) -> str:
        candidates = self.lemmatize_candidates(token)
        return candidates[1] if len(candidates) > 1 else candidates[0]


class NotebookDemoLemmatizer:
    MAPPING = {
        "очен": "очень",
        "очень": "очень",
        "рад": "радость",
        "радость": "радость",
        "счастье": "счастье",
        "грустно": "грусть",
        "грусть": "грусть",
        "страшно": "страх",
        "страх": "страх",
        "зол": "злость",
        "злость": "злость",
        "спокоен": "спокойный",
        "спокойный": "спокойный",
        "ужасный": "ужасный",
        "удивление": "удивление",
    }

    def lemmatize_candidates(self, token: str) -> list[str]:
        lemma = self.MAPPING.get(token, token)
        return [token] if lemma == token else [token, lemma]

    def lemmatize(self, token: str) -> str:
        return self.MAPPING.get(token, token)


def _entry_weight(entry: LexiconEntry) -> float:
    category = entry.source_category.upper()
    category_bonus = {"A": 0.2, "AB": 0.1, "B": 0.0}.get(category, 0.0)
    count_bonus = min(max(entry.source_count, 1), 5) * 0.1
    return 1.0 + category_bonus + count_bonus


def _clause_multipliers(tokens: list[str], config: AnalyzerConfig) -> list[float]:
    if not any(token in config.contrast_markers for token in tokens):
        return [1.0] * len(tokens)
    multipliers = [1.0] * len(tokens)
    contrast_positions = [index for index, token in enumerate(tokens) if token in config.contrast_markers]
    first_contrast = contrast_positions[0]
    for index in range(first_contrast):
        multipliers[index] = config.pre_contrast_discount
    for contrast_index in contrast_positions:
        for index in range(contrast_index + 1, len(tokens)):
            multipliers[index] = config.contrast_boost
    return multipliers


class EmotionAnalyzer:
    def __init__(self, lexicon: EmotionLexicon, *, lemmatizer=None, corrector: TrigramSpellCorrector | None = None, config: AnalyzerConfig | None = None) -> None:
        self.lexicon = lexicon
        self.corrector = corrector
        self.config = config or AnalyzerConfig(collapse_yo=lexicon.collapse_yo)
        if lemmatizer is not None:
            self.lemmatizer = lemmatizer
        elif PYMORPHY3_AVAILABLE:
            self.lemmatizer = Pymorphy3Lemmatizer(collapse_yo=self.config.collapse_yo)
        else:
            self.lemmatizer = NotebookDemoLemmatizer()

    def _tokenize(self, text: str) -> list[str]:
        return [normalize_text(match.group(0), collapse_yo=self.config.collapse_yo) for match in TOKEN_RE.finditer(text)]

    def _correct_token(self, token: str) -> str:
        if not self.corrector or len(token) < self.config.correction_min_length:
            return token
        return self.corrector.correct_token(token)

    def _lemma_candidates(self, token: str) -> list[str]:
        if hasattr(self.lemmatizer, "lemmatize_candidates"):
            raw_candidates = self.lemmatizer.lemmatize_candidates(token)
        else:
            raw_candidates = [self.lemmatizer.lemmatize(token)]

        candidates: list[str] = []
        for candidate in raw_candidates:
            normalized_candidate = normalize_text(candidate, collapse_yo=self.config.collapse_yo)
            if normalized_candidate and normalized_candidate not in candidates:
                candidates.append(normalized_candidate)

        normalized_token = normalize_text(token, collapse_yo=self.config.collapse_yo)
        if normalized_token and normalized_token not in candidates:
            candidates.insert(0, normalized_token)
        return candidates or [normalized_token]

    def _lookup_single_token(self, token: str):
        candidates = self._lemma_candidates(token)
        preferred_lemma = candidates[1] if len(candidates) > 1 else candidates[0]
        for candidate in candidates:
            entries = self.lexicon.lookup(candidate)
            if entries:
                return entries, candidate, candidates
        return [], preferred_lemma, candidates

    def _find_phrase_match(self, tokens: list[str], lemmas: list[str], start_index: int, consumed_until: int):
        if start_index < consumed_until:
            return consumed_until, None
        max_len = min(self.lexicon.max_phrase_len, len(tokens) - start_index)
        for phrase_len in range(max_len, 1, -1):
            surface_candidate = " ".join(tokens[start_index : start_index + phrase_len])
            entries = self.lexicon.lookup(surface_candidate)
            if entries:
                return start_index + phrase_len, entries[0]
            lemma_candidate = " ".join(lemmas[start_index : start_index + phrase_len])
            entries = self.lexicon.lookup(lemma_candidate)
            if entries:
                return start_index + phrase_len, entries[0]
        return start_index + 1, None

    def analyze(self, text: str) -> AnalysisResult:
        raw_tokens = self._tokenize(text)
        corrected_tokens = [self._correct_token(token) for token in raw_tokens]
        lemma_candidates = [self._lemma_candidates(token) for token in corrected_tokens]
        lemmas = [candidates[1] if len(candidates) > 1 else candidates[0] for candidates in lemma_candidates]
        clause_multipliers = _clause_multipliers(corrected_tokens, self.config)
        enough_text = len(corrected_tokens) >= self.config.min_content_tokens
        emotion_scores: dict[str, float] = {}
        matched: list[MatchExplanation] = []
        consumed_until = 0

        for index, token in enumerate(corrected_tokens):
            next_index, phrase_entry = self._find_phrase_match(corrected_tokens, lemmas, index, consumed_until)
            if phrase_entry is not None:
                weight = _entry_weight(phrase_entry) * clause_multipliers[index]
                emotion_scores[phrase_entry.emotion] = emotion_scores.get(phrase_entry.emotion, 0.0) + weight
                matched.append(
                    MatchExplanation(
                        " ".join(raw_tokens[index:next_index]),
                        " ".join(corrected_tokens[index:next_index]),
                        " ".join(corrected_tokens[index:next_index]),
                        " ".join(lemmas[index:next_index]),
                        phrase_entry.term,
                        phrase_entry.emotion,
                        weight,
                        False,
                        False,
                        phrase_entry.source_category,
                        phrase_entry.source_count,
                    )
                )
                consumed_until = next_index
                continue

            if index < consumed_until:
                continue

            entries, matched_lemma, _ = self._lookup_single_token(token)
            if not entries:
                continue
            entry = entries[0]
            weight = _entry_weight(entry) * clause_multipliers[index]
            left_context = lemmas[max(0, index - self.config.negation_window) : index]
            short_context = lemmas[max(0, index - self.config.intensifier_window) : index]
            negated = any(context_token in self.config.negations for context_token in left_context)
            intensified = any(context_token in self.config.intensifiers for context_token in short_context)
            down_toned = any(context_token in self.config.downtoners for context_token in short_context)
            if intensified:
                weight *= self.config.intensity_multiplier
            if down_toned:
                weight *= self.config.downtone_multiplier
            if negated:
                weight *= self.config.negation_multiplier
            emotion_scores[entry.emotion] = emotion_scores.get(entry.emotion, 0.0) + weight
            matched.append(
                MatchExplanation(
                    raw_tokens[index],
                    token,
                    token,
                    matched_lemma,
                    entry.term,
                    entry.emotion,
                    weight,
                    negated,
                    intensified,
                    entry.source_category,
                    entry.source_count,
                )
            )

        positive_score = sum(score for emotion, score in emotion_scores.items() if emotion in POSITIVE_EMOTIONS)
        negative_score = sum(score for emotion, score in emotion_scores.items() if emotion in NEGATIVE_EMOTIONS)
        polarity_score = positive_score - negative_score
        dominant_emotion = None
        if emotion_scores:
            top_emotion, top_score = max(emotion_scores.items(), key=lambda item: item[1])
            if top_score > 0:
                dominant_emotion = top_emotion

        return AnalysisResult(
            original_text=text,
            normalized_text=" ".join(raw_tokens),
            corrected_text=" ".join(corrected_tokens),
            content_token_count=len(corrected_tokens),
            enough_text=enough_text,
            emotion_scores={emotion: round(score, 4) for emotion, score in sorted(emotion_scores.items())},
            polarity_score=round(polarity_score, 4),
            dominant_emotion=dominant_emotion,
            matched=tuple(matched),
        )


def build_emotion_distribution(result: AnalysisResult) -> tuple[dict[str, float], dict[str, float], str | None]:
    positive_scores = {emotion: max(result.emotion_scores.get(emotion, 0.0), 0.0) for emotion in EMOTION_ORDER}
    total = sum(positive_scores.values())
    distribution = {emotion: round((positive_scores[emotion] / total) if total else 0.0, 4) for emotion in EMOTION_ORDER}
    dominant = max(distribution, key=distribution.get) if total else None
    return positive_scores, distribution, dominant


def build_emotion_trace(result: AnalysisResult) -> tuple[list[str], dict[str, list[float]]]:
    tokens = result.corrected_text.split()
    traces = {emotion: [0.0] * len(tokens) for emotion in EMOTION_ORDER}

    if not tokens:
        return tokens, traces

    cursor = 0
    for item in result.matched:
        span_tokens = item.corrected_token.split()
        if not span_tokens:
            continue

        span_len = len(span_tokens)
        found_start = None

        for start in range(cursor, len(tokens) - span_len + 1):
            if tokens[start : start + span_len] == span_tokens:
                found_start = start
                break

        if found_start is None:
            for start in range(0, len(tokens) - span_len + 1):
                if tokens[start : start + span_len] == span_tokens:
                    found_start = start
                    break

        if found_start is None:
            continue

        share = item.weight / span_len
        for offset in range(span_len):
            traces[item.emotion][found_start + offset] += share

        cursor = found_start + span_len

    return tokens, traces


def safe_torch_load(path: Path, map_location: torch.device):
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "CharacterMapper"):
        setattr(main_module, "CharacterMapper", CharacterMapper)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def infer_crnn_config_from_state_dict(state_dict: dict, mapper: CharacterMapper) -> dict[str, int]:
    hidden_size = 256
    num_layers = 2

    map_weight = state_dict.get("map_to_seq.weight")
    if map_weight is not None and hasattr(map_weight, "shape") and len(map_weight.shape) >= 1:
        hidden_size = int(map_weight.shape[0])

    fc_weight = state_dict.get("fc.weight")
    if fc_weight is not None and hasattr(fc_weight, "shape") and len(fc_weight.shape) >= 2:
        hidden_size = int(fc_weight.shape[1] // 2)
        expected_outputs = mapper.num_classes + 1
        actual_outputs = int(fc_weight.shape[0])
        if actual_outputs != expected_outputs:
            raise RuntimeError(
                f"Размер выходного слоя файла модели ({actual_outputs}) не совпадает с размером mapper ({expected_outputs}). "
                "Выбранный файл модели и его char_mapper несовместимы."
            )

    layer_indices = []
    for key in state_dict:
        match = re.fullmatch(r"rnn\.weight_ih_l(\d+)(?:_reverse)?", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if layer_indices:
        num_layers = max(layer_indices) + 1

    return {
        "img_height": 64,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }


@dataclass
class OCRModelBundle:
    language: str
    checkpoint_path: Path
    model: CRNN
    mapper: CharacterMapper
    device: torch.device
    img_height: int = 64

    def resize_for_model(self, img: Image.Image, *, resample: Image.Resampling = Image.Resampling.LANCZOS) -> Image.Image:
        _, resized_img, _ = build_tensor_from_pil(img, img_height=self.img_height, resample=resample)
        return resized_img

    def predict_resized_pil(self, resized_img: Image.Image) -> str:
        arr = np.array(resized_img, dtype=np.float32) / 255.0
        arr_norm = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        return decode_predictions(outputs, self.mapper)[0]

    def predict_pil(self, img: Image.Image, *, resample: Image.Resampling = Image.Resampling.LANCZOS) -> tuple[Image.Image, str]:
        resized_img = self.resize_for_model(img, resample=resample)
        return resized_img, self.predict_resized_pil(resized_img)


@dataclass
class OCRVariantResult:
    key: str
    label: str
    prediction: str
    preview_image: Image.Image
    chunk_strip_image: Image.Image
    chunk_predictions: list[str]
    word_bounds: list[tuple[int, int]]
    chunk_bounds: list[tuple[int, int]]
    notes: list[str]


@dataclass
class OCRResult:
    image_path: Path
    language: str
    threshold_enabled: bool
    checkpoint_path: Path
    prediction: str
    original_preview: Image.Image
    processed_preview: Image.Image
    model_input_preview: Image.Image
    chunk_predictions: list[str]
    word_bounds: list[tuple[int, int]]
    chunk_bounds: list[tuple[int, int]]
    notes: list[str]


@dataclass
class EmotionSummary:
    available: bool
    note: str
    result: AnalysisResult | None = None
    distribution: dict[str, float] | None = None
    dominant: str | None = None


@dataclass
class VariantAnalysis:
    ocr: OCRVariantResult
    emotion: EmotionSummary


@dataclass
class AnalysisBundle:
    image_path: Path
    language: str
    checkpoint_path: Path
    original_preview: Image.Image
    variants: dict[str, VariantAnalysis]


class EmotionRuntime:
    def __init__(self, analyzer: EmotionAnalyzer, resource_note: str) -> None:
        self.analyzer = analyzer
        self.resource_note = resource_note

    @classmethod
    def create_default(cls) -> "EmotionRuntime":
        if WORDFREQ_FREQ_PATH.exists():
            corrector = TrigramSpellCorrector.from_frequency_file(WORDFREQ_FREQ_PATH)
            corrector_note = f"Частотный словарь загружен из {WORDFREQ_FREQ_PATH.name}"
        elif WORDFREQ_AVAILABLE:
            word_frequencies = build_word_frequencies_from_wordfreq()
            save_word_frequencies(word_frequencies, WORDFREQ_FREQ_PATH)
            corrector = TrigramSpellCorrector(word_frequencies)
            corrector_note = f"Частотный словарь собран из wordfreq: {len(word_frequencies)} слов"
        else:
            demo_frequencies = Counter(
                {
                    "я": 500,
                    "мне": 250,
                    "это": 300,
                    "очень": 220,
                    "этому": 150,
                    "дню": 120,
                    "чувствую": 80,
                    "сегодня": 170,
                    "результат": 160,
                    "скорее": 140,
                    "слегка": 90,
                    "в": 400,
                    "целом": 110,
                    "и": 600,
                    "не": 500,
                    "но": 450,
                    "рад": 120,
                    "радость": 100,
                    "счастье": 80,
                    "грустно": 90,
                    "грусть": 70,
                    "страшно": 85,
                    "страх": 60,
                    "зол": 75,
                    "злость": 50,
                    "удивление": 40,
                    "спокоен": 65,
                    "спокойный": 25,
                    "ужасный": 20,
                    "прекрасный": 20,
                }
            )
            corrector = TrigramSpellCorrector(demo_frequencies)
            corrector_note = "Используется небольшой демонстрационный словарь"

        if LEXICON_PATH.exists():
            lexicon = load_rusemolex(LEXICON_PATH)
            lexicon_note = f"RusEmoLex загружен из {LEXICON_PATH.name}"
        else:
            lexicon = EmotionLexicon(
                [
                    LexiconEntry(term="радость", emotion="joy", source_category="A", source_count=4),
                    LexiconEntry(term="счастье", emotion="joy", source_category="A", source_count=3),
                    LexiconEntry(term="грусть", emotion="sadness", source_category="AB", source_count=2),
                    LexiconEntry(term="страх", emotion="fear", source_category="A", source_count=3),
                    LexiconEntry(term="злость", emotion="anger", source_category="A", source_count=3),
                    LexiconEntry(term="ужасный", emotion="disgust", source_category="AB", source_count=2),
                    LexiconEntry(term="удивление", emotion="surprise", source_category="AB", source_count=2),
                    LexiconEntry(term="спокойный", emotion="joy", source_category="AB", source_count=1),
                ]
            )
            lexicon_note = "Используется небольшой демонстрационный лексикон"

        if PYMORPHY3_AVAILABLE:
            analyzer = EmotionAnalyzer(lexicon, lemmatizer=Pymorphy3Lemmatizer(), corrector=corrector)
            lemmatizer_note = "Лемматизация pymorphy3 включена"
        else:
            analyzer = EmotionAnalyzer(lexicon, lemmatizer=NotebookDemoLemmatizer(), corrector=corrector)
            lemmatizer_note = "Используется демонстрационный лемматизатор"

        resource_note = "; ".join([lexicon_note, corrector_note, lemmatizer_note])
        return cls(analyzer=analyzer, resource_note=resource_note)

    def analyze(self, text: str) -> EmotionSummary:
        result = self.analyzer.analyze(text)
        _, distribution, dominant = build_emotion_distribution(result)
        return EmotionSummary(True, self.resource_note, result, distribution, dominant)


class InferenceEngine:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_cache: dict[str, OCRModelBundle] = {}
        self._emotion_runtime: EmotionRuntime | None = None

    def _checkpoint_candidates(self, language: str) -> list[Path]:
        if language == "ru":
            return [ROOT / "crnn_ru_checkpoint_last.pth", ROOT / "trained_models" / "crnn_checkpoint_best_ru.pth"]
        return [ROOT / "trained_models" / "crnn_checkpoint_best_en.pth"]

    def list_available_checkpoints(self) -> list[Path]:
        roots = [ROOT, ROOT / "trained_models"]
        paths: list[Path] = []
        for base in roots:
            if not base.exists():
                continue
            for path in base.rglob("*.pth"):
                if path.is_file() and path not in paths:
                    paths.append(path)
        return sorted(paths, key=lambda item: str(item.relative_to(ROOT)).lower())

    def list_available_checkpoint_labels(self) -> list[str]:
        return [str(path.relative_to(ROOT)) for path in self.list_available_checkpoints()]

    def default_checkpoint_label(self, language: str) -> str:
        labels = self.list_available_checkpoint_labels()
        if not labels:
            return ""
        for path in self._checkpoint_candidates(language):
            if path.exists():
                try:
                    label = str(path.relative_to(ROOT))
                except ValueError:
                    label = str(path)
                if label in labels:
                    return label
        preferred_tokens = (
            ["_ru", "ru_", "best_ru", "checkpoint_best_ru"]
            if language == "ru"
            else ["_en", "en_", "best_en", "checkpoint_best_en"]
        )
        for label in labels:
            lower = label.lower()
            if any(token in lower for token in preferred_tokens):
                return label
        return labels[0]

    def resolve_checkpoint_path(self, checkpoint_label: str | None, language: str) -> Path:
        if checkpoint_label:
            path = Path(checkpoint_label)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                return path

        default_label = self.default_checkpoint_label(language)
        if default_label:
            return ROOT / default_label

        checkpoint_path = next((path for path in self._checkpoint_candidates(language) if path.exists()), None)
        if checkpoint_path is not None:
            return checkpoint_path
        raise FileNotFoundError(f"Не найден файл модели для языка={language}")

    def get_model(self, language: str, checkpoint_label: str | None = None) -> OCRModelBundle:
        checkpoint_path = self.resolve_checkpoint_path(checkpoint_label, language)
        cache_key = str(checkpoint_path.resolve()).lower()
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if checkpoint_path is None:
            raise FileNotFoundError(f"Не найден файл модели для языка={language}")

        checkpoint = safe_torch_load(checkpoint_path, self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            mapper = checkpoint.get("char_mapper")
        else:
            state_dict = checkpoint
            mapper = None

        if mapper is None:
            raise RuntimeError(f"Файл модели {checkpoint_path.name} не содержит char_mapper")

        model_config = infer_crnn_config_from_state_dict(state_dict, mapper)
        model = CRNN(
            img_height=model_config["img_height"],
            num_chars=mapper.num_classes,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
        ).to(self.device)
        with torch.no_grad():
            _ = model(torch.randn(1, 1, model_config["img_height"], 256).to(self.device))
        model.load_state_dict(state_dict)
        model.eval()

        bundle = OCRModelBundle(language, checkpoint_path, model, mapper, self.device, img_height=model_config["img_height"])
        self._model_cache[cache_key] = bundle
        return bundle

    def get_emotion_runtime(self) -> EmotionRuntime:
        if self._emotion_runtime is None:
            self._emotion_runtime = EmotionRuntime.create_default()
        return self._emotion_runtime

    def preprocess_baseline(self, image_path: Path, img_height: int = 64) -> dict[str, object]:
        img_original = Image.open(image_path).convert("L")
        img_enhanced = enhance_handwriting_image(img_original)
        img_cropped, cropped_binary, crop_bbox = crop_to_content(img_enhanced)
        _, img_resized, _ = build_tensor_from_pil(img_cropped, img_height=img_height)
        return {
            "original": img_original,
            "enhanced": img_enhanced,
            "cropped": img_cropped,
            "resized": img_resized,
            "cropped_binary": cropped_binary,
            "crop_bbox": crop_bbox,
        }

    def make_thresholded_preprocessing(
        self,
        image_path: Path,
        *,
        img_height: int = 64,
        threshold_value: int = MASK_THRESHOLD,
        min_component_area: int = MIN_COMPONENT_AREA,
    ) -> dict[str, object]:
        baseline = self.preprocess_baseline(image_path, img_height=img_height)
        working_img = baseline["enhanced"]

        if THRESHOLD_UPSCALE_FACTOR > 1:
            working_img = working_img.resize(
                (working_img.width * THRESHOLD_UPSCALE_FACTOR, working_img.height * THRESHOLD_UPSCALE_FACTOR),
                Image.Resampling.LANCZOS,
            )

        median_size = THRESHOLD_MEDIAN_SIZE + (THRESHOLD_MEDIAN_SIZE % 2 == 0)
        if median_size > 1:
            working_img = working_img.filter(ImageFilter.MedianFilter(size=median_size))

        working_arr = np.array(working_img, dtype=np.uint8)
        binary_arr = np.where(working_arr > threshold_value, 255, 0).astype(np.uint8)
        scaled_min_area = min_component_area * max(1, THRESHOLD_UPSCALE_FACTOR) * max(1, THRESHOLD_UPSCALE_FACTOR)
        cleaned_arr = remove_small_black_components(binary_arr, min_area=scaled_min_area)
        thresholded_upscaled_img = Image.fromarray(cleaned_arr)

        content_bbox = compute_content_bbox_from_binary(cleaned_arr, padding=CONTENT_CROP_PADDING)
        thresholded_cropped_img = thresholded_upscaled_img.crop(content_bbox)
        thresholded_cropped_img, tight_bbox = crop_binary_pil_to_content(
            thresholded_cropped_img,
            padding=max(1, CONTENT_CROP_PADDING // 2),
        )
        _, thresholded_model_img, _ = build_tensor_from_pil(
            thresholded_cropped_img,
            img_height=img_height,
            resample=Image.Resampling.NEAREST,
        )
        return {
            "baseline": baseline,
            "working_img": working_img,
            "thresholded_upscaled_img": thresholded_upscaled_img,
            "thresholded_cropped_img": thresholded_cropped_img,
            "thresholded_model_img": thresholded_model_img,
            "content_bbox": content_bbox,
            "tight_bbox": tight_bbox,
        }

    def segment_thresholded_words(self, thresholded_img: Image.Image) -> dict[str, object]:
        binary_arr = np.array(thresholded_img, dtype=np.uint8)
        word_bounds = find_word_bounds(binary_arr)
        chunk_bounds = group_word_bounds(word_bounds, max_width=thresholded_img.width)
        chunk_images = []
        for start, end in chunk_bounds:
            raw_chunk = thresholded_img.crop((start, 0, end, thresholded_img.height))
            cropped_chunk, _ = crop_binary_pil_to_content(raw_chunk, padding=max(1, WORD_CHUNK_PADDING // 2))
            chunk_images.append(cropped_chunk)
        return {
            "binary_arr": binary_arr,
            "word_bounds": word_bounds,
            "chunk_bounds": chunk_bounds,
            "chunk_images": chunk_images,
        }

    def build_emotion_summary(self, language: str, prediction: str) -> EmotionSummary:
        if language != "ru":
            return EmotionSummary(False, "Анализ эмоций в этом репозитории сейчас реализован только для русского текста.")
        return self.get_emotion_runtime().analyze(prediction)

    def _predict_segmented_variant(
        self,
        *,
        bundle: OCRModelBundle,
        source_img: Image.Image,
        binary_arr: np.ndarray,
        variant_key: str,
        variant_label: str,
        chunk_resample: Image.Resampling,
        note_prefix: str,
    ) -> OCRVariantResult:
        segmentation = segment_words_from_source(source_img, binary_arr)
        chunk_predictions: list[str] = []
        for chunk_img in segmentation["chunk_images"]:
            _, chunk_prediction = bundle.predict_pil(chunk_img, resample=chunk_resample)
            chunk_predictions.append(chunk_prediction.strip())

        prediction = " ".join(pred for pred in chunk_predictions if pred).strip()
        if not prediction:
            fallback_preview, prediction = bundle.predict_pil(source_img, resample=chunk_resample)
            chunk_predictions = [prediction]
            chunk_strip_image = fallback_preview
            chunk_bounds = [(0, source_img.width)]
            word_bounds = [(0, source_img.width)]
        else:
            chunk_strip_image = stitch_chunk_previews(segmentation["chunk_images"] or [source_img])
            chunk_bounds = segmentation["chunk_bounds"]
            word_bounds = segmentation["word_bounds"]

        preview_image = draw_segmented_preview(source_img, chunk_bounds)
        return OCRVariantResult(
            key=variant_key,
            label=variant_label,
            prediction=prediction,
            preview_image=preview_image,
            chunk_strip_image=chunk_strip_image,
            chunk_predictions=chunk_predictions,
            word_bounds=word_bounds,
            chunk_bounds=chunk_bounds,
            notes=[
                f"{note_prefix}: распознано чанков: {len(chunk_predictions)}",
            ],
        )

    def analyze_image_bundle(
        self,
        image_path: Path,
        *,
        language: str,
        checkpoint_label: str | None = None,
        threshold_value: int = MASK_THRESHOLD,
        min_component_area: int = MIN_COMPONENT_AREA,
    ) -> AnalysisBundle:
        bundle = self.get_model(language, checkpoint_label=checkpoint_label)
        baseline = self.preprocess_baseline(image_path, img_height=bundle.img_height)
        baseline_model_img = baseline["resized"]
        baseline_prediction = bundle.predict_resized_pil(baseline_model_img)
        baseline_ocr = OCRVariantResult(
            key="baseline",
            label="Базовая",
            prediction=baseline_prediction,
            preview_image=baseline_model_img,
            chunk_strip_image=baseline_model_img,
            chunk_predictions=[baseline_prediction],
            word_bounds=[],
            chunk_bounds=[],
            notes=[
                "Базовая: один проход после масштабирования",
                f"Модель: {bundle.checkpoint_path.name}",
                f"Порог маски={threshold_value}, мин. площадь={min_component_area}",
            ],
        )

        baseline_binary = build_binary_mask(baseline["cropped"], threshold=threshold_value)
        baseline_segmented_ocr = self._predict_segmented_variant(
            bundle=bundle,
            source_img=baseline["cropped"],
            binary_arr=baseline_binary,
            variant_key="baseline_segmented",
            variant_label="Базовая + чанки",
            chunk_resample=Image.Resampling.LANCZOS,
            note_prefix="Базовая с чанками",
        )
        baseline_segmented_ocr.notes.append(f"Порог маски={threshold_value}, мин. площадь={min_component_area}")

        thresholded = self.make_thresholded_preprocessing(
            image_path,
            img_height=bundle.img_height,
            threshold_value=threshold_value,
            min_component_area=min_component_area,
        )
        threshold_binary = np.array(thresholded["thresholded_cropped_img"], dtype=np.uint8)
        thresholded_segmented_ocr = self._predict_segmented_variant(
            bundle=bundle,
            source_img=thresholded["thresholded_cropped_img"],
            binary_arr=threshold_binary,
            variant_key="thresholded_segmented",
            variant_label="Бинаризация + чанки",
            chunk_resample=Image.Resampling.NEAREST,
            note_prefix="Бинаризация с чанками",
        )
        thresholded_segmented_ocr.notes.append(f"Порог маски={threshold_value}, мин. площадь={min_component_area}")

        return AnalysisBundle(
            image_path=image_path,
            language=language,
            checkpoint_path=bundle.checkpoint_path,
            original_preview=baseline["original"],
            variants={
                "baseline": VariantAnalysis(baseline_ocr, self.build_emotion_summary(language, baseline_ocr.prediction)),
                "baseline_segmented": VariantAnalysis(
                    baseline_segmented_ocr,
                    self.build_emotion_summary(language, baseline_segmented_ocr.prediction),
                ),
                "thresholded_segmented": VariantAnalysis(
                    thresholded_segmented_ocr,
                    self.build_emotion_summary(language, thresholded_segmented_ocr.prediction),
                ),
            },
        )

    def analyze_image(
        self,
        image_path: Path,
        *,
        language: str,
        threshold_enabled: bool,
        threshold_value: int = MASK_THRESHOLD,
        min_component_area: int = MIN_COMPONENT_AREA,
    ) -> tuple[OCRResult, EmotionSummary]:
        bundle = self.get_model(language)
        baseline = self.preprocess_baseline(image_path, img_height=bundle.img_height)
        notes = [f"Устройство: {bundle.device.type}", f"Модель: {bundle.checkpoint_path.name}"]

        if threshold_enabled:
            thresholded = self.make_thresholded_preprocessing(
                image_path,
                img_height=bundle.img_height,
                threshold_value=threshold_value,
                min_component_area=min_component_area,
            )
            segmented = self.segment_thresholded_words(thresholded["thresholded_cropped_img"])

            chunk_predictions = []
            for chunk_img in segmented["chunk_images"]:
                _, chunk_prediction = bundle.predict_pil(chunk_img, resample=Image.Resampling.NEAREST)
                chunk_predictions.append(chunk_prediction.strip())

            prediction = " ".join(pred for pred in chunk_predictions if pred)
            if not prediction:
                prediction = bundle.predict_resized_pil(thresholded["thresholded_model_img"])
                chunk_predictions = [prediction]
                segmented["word_bounds"] = [(0, thresholded["thresholded_cropped_img"].width)]
                segmented["chunk_bounds"] = [(0, thresholded["thresholded_cropped_img"].width)]

            processed_preview = draw_word_bound_preview(thresholded["thresholded_cropped_img"], segmented["word_bounds"])
            model_input_preview = stitch_chunk_previews(segmented["chunk_images"] or [thresholded["thresholded_cropped_img"]])
            notes.append(
                f"Бинаризация включена: границ слов={len(segmented['word_bounds'])}, чанков={len(segmented['chunk_bounds'])}"
            )
            notes.append(f"Порог маски={threshold_value}, мин. площадь={min_component_area}")
            ocr_result = OCRResult(
                image_path=image_path,
                language=language,
                threshold_enabled=True,
                checkpoint_path=bundle.checkpoint_path,
                prediction=prediction,
                original_preview=baseline["original"],
                processed_preview=processed_preview,
                model_input_preview=model_input_preview,
                chunk_predictions=chunk_predictions,
                word_bounds=list(segmented["word_bounds"]),
                chunk_bounds=list(segmented["chunk_bounds"]),
                notes=notes,
            )
        else:
            baseline_model_img, prediction = bundle.predict_pil(baseline["cropped"])
            notes.append("Бинаризация выключена: один базовый проход")
            notes.append(f"Порог маски={threshold_value}, мин. площадь={min_component_area}")
            ocr_result = OCRResult(
                image_path=image_path,
                language=language,
                threshold_enabled=False,
                checkpoint_path=bundle.checkpoint_path,
                prediction=prediction,
                original_preview=baseline["original"],
                processed_preview=baseline["cropped"],
                model_input_preview=baseline_model_img,
                chunk_predictions=[prediction],
                word_bounds=[],
                chunk_bounds=[],
                notes=notes,
            )

        if language != "ru":
            emotion_summary = EmotionSummary(False, "Анализ эмоций в этом репозитории сейчас реализован только для русского текста.")
        else:
            emotion_summary = self.get_emotion_runtime().analyze(ocr_result.prediction)
        return ocr_result, emotion_summary


def format_distribution(distribution: dict[str, float] | None) -> str:
    if not distribution:
        return "н/д"
    return ", ".join(f"{format_emotion_label(emotion)}={distribution.get(emotion, 0.0):.2f}" for emotion in EMOTION_ORDER)


def format_matches(result: AnalysisResult | None, limit: int = 8) -> str:
    if result is None or not result.matched:
        return "нет"
    lines = []
    for item in result.matched[:limit]:
        flags = []
        if item.intensified:
            flags.append("усиление")
        if item.negated:
            flags.append("отрицание")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"{item.original_token} -> {item.lemma} -> {item.matched_term} -> "
            f"{format_emotion_label(item.emotion)} ({item.weight:.2f}){flag_text}"
        )
    if len(result.matched) > limit:
        lines.append(f"... и еще {len(result.matched) - limit}")
    return "\n".join(lines)


def make_preview_photo(image: Image.Image, max_size: tuple[int, int] = PREVIEW_MAX_SIZE) -> ImageTk.PhotoImage:
    preview = ImageOps.contain(image.copy(), max_size, Image.Resampling.LANCZOS)
    if preview.mode not in {"RGB", "RGBA", "L"}:
        preview = preview.convert("RGB")
    return ImageTk.PhotoImage(preview)


def make_emotion_graph_image(
    emotion_summary: EmotionSummary,
    *,
    width: int = GRAPH_SIZE[0],
    height: int = GRAPH_SIZE[1],
) -> Image.Image:
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)

    if not emotion_summary.available or emotion_summary.result is None:
        ax.axis("off")
        ax.text(0.02, 0.62, "Анализ эмоций недоступен", fontsize=14, transform=ax.transAxes)
        ax.text(0.02, 0.42, emotion_summary.note, fontsize=10, color="#666666", transform=ax.transAxes)
    else:
        result = emotion_summary.result
        _, distribution, dominant = build_emotion_distribution(result)
        tokens, traces = build_emotion_trace(result)

        if not tokens:
            ax.axis("off")
            ax.text(0.02, 0.62, "Нет токенов для графика", fontsize=14, transform=ax.transAxes)
            ax.text(0.02, 0.42, emotion_summary.note, fontsize=10, color="#666666", transform=ax.transAxes)
        else:
            x = list(range(1, len(tokens) + 1))
            for emotion in EMOTION_ORDER:
                linewidth = 3.6 if emotion == dominant else 1.8
                alpha = 1.0 if emotion == dominant else 0.75
                ax.plot(
                    x,
                    traces[emotion],
                    marker="o",
                    linewidth=linewidth,
                    alpha=alpha,
                    color=EMOTION_COLORS[emotion],
                    label=f"{format_emotion_label(emotion)} ({distribution[emotion]:.2f})",
                )

            ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{idx}:{token}" for idx, token in zip(x, tokens)], rotation=45, ha="right")
            ax.set_xlabel("Номер слова")
            ax.set_ylabel("Вклад эмоции")
            dominant_label = format_emotion_label(dominant)
            ax.set_title(f"Динамика эмоций | доминирует: {dominant_label}")
            legend = ax.legend(title="Распределение эмоций")

            if dominant is not None:
                for text in legend.get_texts():
                    if text.get_text().startswith(dominant_label):
                        text.set_fontweight("bold")

    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("RGB")


class OCRUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Распознавание рукописного текста и эмоций")
        self.root.geometry("1650x980")
        self.root.minsize(1300, 860)

        self.engine = InferenceEngine()
        self.file_var = tk.StringVar(value=self._default_image_path())
        self.language_var = tk.StringVar(value="ru")
        self.checkpoint_var = tk.StringVar()
        self.graph_variant_var = tk.StringVar(value="thresholded_segmented")
        self.mask_threshold_var = tk.IntVar(value=MASK_THRESHOLD)
        self.min_area_var = tk.IntVar(value=MIN_COMPONENT_AREA)
        self.status_var = tk.StringVar(value="Готово")
        self.summary_var = tk.StringVar(value="Выберите изображение и нажмите «Анализировать».")
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.last_bundle: AnalysisBundle | None = None

        self._build_layout()
        self.refresh_checkpoint_options(force_default=True)
        placeholder_summary = EmotionSummary(False, "Запустите анализ, чтобы увидеть график эмоций.")
        self._set_preview("graph", self.graph_label, make_emotion_graph_image(placeholder_summary), max_size=GRAPH_SIZE)

    def _default_image_path(self) -> str:
        if DEFAULT_EXAMPLES_DIR.exists():
            for name in ("udivlen_big.jpg", "problems.jpg", "IMG_20260409_215827.jpg"):
                candidate = DEFAULT_EXAMPLES_DIR / name
                if candidate.exists():
                    return str(candidate)
            for candidate in sorted(DEFAULT_EXAMPLES_DIR.iterdir()):
                if candidate.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    return str(candidate)
        return ""

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="Изображение").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.file_var).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(controls, text="Выбрать...", command=self.browse_file).grid(row=0, column=2, padx=(0, 12))
        ttk.Label(controls, text="Язык").grid(row=0, column=3, sticky="w", padx=(0, 8))
        self.language_combo = ttk.Combobox(controls, textvariable=self.language_var, values=["ru", "en"], state="readonly", width=8)
        self.language_combo.grid(row=0, column=4, padx=(0, 12))
        self.language_combo.bind("<<ComboboxSelected>>", self.on_language_changed)
        ttk.Label(controls, text="Модель (.pth)").grid(row=0, column=5, sticky="w", padx=(0, 8))
        self.checkpoint_combo = ttk.Combobox(controls, textvariable=self.checkpoint_var, state="readonly", width=40)
        self.checkpoint_combo.grid(row=0, column=6, sticky="ew", padx=(0, 12))
        ttk.Button(controls, text="Анализировать", command=self.analyze).grid(row=0, column=7)

        ttk.Label(controls, text="Порог маски").grid(row=1, column=0, sticky="w", pady=(10, 0))
        tk.Scale(
            controls,
            variable=self.mask_threshold_var,
            from_=120,
            to=245,
            orient="horizontal",
            resolution=1,
            length=260,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(controls, text="Мин. площадь").grid(row=1, column=3, sticky="w", pady=(10, 0))
        tk.Scale(
            controls,
            variable=self.min_area_var,
            from_=0,
            to=80,
            orient="horizontal",
            resolution=1,
            length=180,
        ).grid(row=1, column=4, sticky="w", pady=(8, 0))

        ttk.Label(controls, text="Предобработка").grid(row=1, column=5, sticky="w", pady=(10, 0))
        method_switches = ttk.Frame(controls)
        method_switches.grid(row=1, column=6, sticky="w", pady=(8, 0))
        for key, label in VARIANT_ORDER:
            ttk.Radiobutton(
                method_switches,
                text=label,
                value=key,
                variable=self.graph_variant_var,
                command=self.refresh_graph,
            ).pack(side="left", padx=(0, 8))

        preview_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        preview_frame.grid(row=1, column=0, sticky="nsew")
        for col in range(4):
            preview_frame.columnconfigure(col, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        self.original_title = ttk.Label(preview_frame, text="Исходное")
        self.original_title.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.baseline_title = ttk.Label(preview_frame, text="Базовая")
        self.baseline_title.grid(row=0, column=1, sticky="w", pady=(0, 6))
        self.baseline_segmented_title = ttk.Label(preview_frame, text="Базовая + чанки")
        self.baseline_segmented_title.grid(row=0, column=2, sticky="w", pady=(0, 6))
        self.thresholded_title = ttk.Label(preview_frame, text="Бинаризация + чанки")
        self.thresholded_title.grid(row=0, column=3, sticky="w", pady=(0, 6))

        self.original_image_label = ttk.Label(preview_frame, relief="solid", anchor="center")
        self.original_image_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.baseline_image_label = ttk.Label(preview_frame, relief="solid", anchor="center")
        self.baseline_image_label.grid(row=1, column=1, sticky="nsew", padx=(0, 8))
        self.baseline_segmented_image_label = ttk.Label(preview_frame, relief="solid", anchor="center")
        self.baseline_segmented_image_label.grid(row=1, column=2, sticky="nsew", padx=(0, 8))
        self.thresholded_image_label = ttk.Label(preview_frame, relief="solid", anchor="center")
        self.thresholded_image_label.grid(row=1, column=3, sticky="nsew")

        output_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        output_frame.grid(row=2, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)

        self.summary_label = ttk.Label(
            output_frame,
            textvariable=self.summary_var,
            justify="left",
            anchor="w",
            wraplength=1250,
        )
        self.summary_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.graph_label = ttk.Label(output_frame, relief="solid", anchor="center")
        self.graph_label.grid(row=1, column=0, sticky="nsew")

        ttk.Label(self.root, textvariable=self.status_var, padding=(12, 6)).grid(row=3, column=0, sticky="ew")
        self.root.bind("<Return>", lambda _event: self.analyze())

    def browse_file(self) -> None:
        start_dir = DEFAULT_EXAMPLES_DIR if DEFAULT_EXAMPLES_DIR.exists() else ROOT
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите изображение с рукописным текстом",
            initialdir=start_dir,
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")],
        )
        if chosen:
            self.file_var.set(chosen)

    def _set_preview(self, slot: str, widget: ttk.Label, image: Image.Image, *, max_size: tuple[int, int] = PREVIEW_MAX_SIZE) -> None:
        photo = make_preview_photo(image, max_size=max_size)
        self._photo_refs[slot] = photo
        widget.configure(image=photo)

    def _short(self, value: str, limit: int = 54) -> str:
        compact = " ".join(value.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def refresh_checkpoint_options(self, *, force_default: bool = False) -> None:
        labels = self.engine.list_available_checkpoint_labels()
        self.checkpoint_combo.configure(values=labels)
        preferred = self.engine.default_checkpoint_label(self.language_var.get())
        current = self.checkpoint_var.get()
        if force_default or current not in labels:
            self.checkpoint_var.set(preferred or (labels[0] if labels else ""))

    def on_language_changed(self, _event=None) -> None:
        self.refresh_checkpoint_options(force_default=True)

    def render_summary(self, bundle: AnalysisBundle) -> str:
        selected_key = self.graph_variant_var.get()
        selected = bundle.variants[selected_key]
        parts = [
            f"Язык: {bundle.language} | Модель: {bundle.checkpoint_path.name}",
            f"Предобработка: {selected.ocr.label} | Порог маски: {self.mask_threshold_var.get()} | Мин. площадь: {self.min_area_var.get()}",
            "",
        ]
        for key, label in VARIANT_ORDER:
            parts.append(f"{label}: {bundle.variants[key].ocr.prediction or '<пусто>'}")
        if selected.emotion.available and selected.emotion.result is not None:
            result = selected.emotion.result
            parts.extend(
                [
                    "",
                    f"Доминирующая эмоция выбранного варианта: {format_emotion_label(selected.emotion.dominant)}",
                    f"Распределение выбранного варианта: {format_distribution(selected.emotion.distribution)}",
                    f"Исправленный текст выбранного варианта: {result.corrected_text or '<пусто>'}",
                ]
            )
        else:
            parts.append(selected.emotion.note)
        return "\n".join(parts)

    def render_cli_output(self, bundle: AnalysisBundle) -> str:
        lines = [
            f"Изображение: {bundle.image_path}",
            f"Язык:        {bundle.language}",
            f"Модель:      {bundle.checkpoint_path}",
            "",
        ]
        for key, label in VARIANT_ORDER:
            variant = bundle.variants[key]
            lines.extend(
                [
                    label.upper(),
                    f"OCR-результат: {variant.ocr.prediction or '<пусто>'}",
                    f"Результаты чанков: {variant.ocr.chunk_predictions}",
                    f"Границы чанков: {variant.ocr.chunk_bounds}",
                ]
            )
            if variant.emotion.available and variant.emotion.result is not None:
                result = variant.emotion.result
                lines.extend(
                    [
                        f"Доминирующая эмоция: {format_emotion_label(variant.emotion.dominant)}",
                        f"Распределение: {format_distribution(variant.emotion.distribution)}",
                        f"Исправленный текст: {result.corrected_text}",
                        f"Совпадения: {format_matches(result)}",
                    ]
                )
            else:
                lines.append(variant.emotion.note)
            lines.append("")
        return "\n".join(lines).rstrip()

    def refresh_graph(self) -> None:
        if self.last_bundle is None:
            placeholder_summary = EmotionSummary(False, "Запустите анализ, чтобы увидеть график эмоций.")
            self.summary_var.set("Выберите изображение и нажмите «Анализировать».")
            self._set_preview("graph", self.graph_label, make_emotion_graph_image(placeholder_summary), max_size=GRAPH_SIZE)
            return

        selected_key = self.graph_variant_var.get()
        if selected_key not in self.last_bundle.variants:
            selected_key = "thresholded_segmented"
            self.graph_variant_var.set(selected_key)
        selected = self.last_bundle.variants[selected_key]
        self.summary_var.set(self.render_summary(self.last_bundle))
        self._set_preview("graph", self.graph_label, make_emotion_graph_image(selected.emotion), max_size=GRAPH_SIZE)

    def analyze(self) -> None:
        raw_path = self.file_var.get().strip()
        if not raw_path:
            messagebox.showerror("Нет изображения", "Сначала выберите изображение.")
            return

        image_path = Path(raw_path)
        if not image_path.exists():
            messagebox.showerror("Файл не найден", f"Файл не существует:\n{image_path}")
            return

        self.status_var.set("Идет анализ...")
        self.root.update_idletasks()
        try:
            bundle = self.engine.analyze_image_bundle(
                image_path=image_path,
                language=self.language_var.get(),
                checkpoint_label=self.checkpoint_var.get(),
                threshold_value=self.mask_threshold_var.get(),
                min_component_area=self.min_area_var.get(),
            )
        except Exception as exc:
            self.status_var.set("Ошибка")
            messagebox.showerror("Анализ не выполнен", str(exc))
            return

        self.last_bundle = bundle
        self.original_title.configure(text=f"Исходное\n{bundle.image_path.name}")
        self._set_preview("original", self.original_image_label, bundle.original_preview, max_size=GRID_PREVIEW_SIZE)
        self._set_preview("baseline", self.baseline_image_label, bundle.variants["baseline"].ocr.preview_image, max_size=GRID_PREVIEW_SIZE)
        self._set_preview(
            "baseline_segmented",
            self.baseline_segmented_image_label,
            bundle.variants["baseline_segmented"].ocr.preview_image,
            max_size=GRID_PREVIEW_SIZE,
        )
        self._set_preview(
            "thresholded_segmented",
            self.thresholded_image_label,
            bundle.variants["thresholded_segmented"].ocr.preview_image,
            max_size=GRID_PREVIEW_SIZE,
        )

        self.baseline_title.configure(text=f"Базовая\n{self._short(bundle.variants['baseline'].ocr.prediction)}")
        self.baseline_segmented_title.configure(
            text=f"Базовая + чанки\n{self._short(bundle.variants['baseline_segmented'].ocr.prediction)}"
        )
        self.thresholded_title.configure(
            text=f"Бинаризация + чанки\n{self._short(bundle.variants['thresholded_segmented'].ocr.prediction)}"
        )
        self.refresh_graph()
        self.status_var.set(
            f"Готово: {self._short(bundle.variants[self.graph_variant_var.get()].ocr.prediction or '<пусто>', limit=70)}"
        )


def run_cli_analysis(
    image_path: Path,
    language: str,
    checkpoint_label: str | None,
    mask_threshold: int,
    min_area: int,
) -> int:
    engine = InferenceEngine()
    bundle = engine.analyze_image_bundle(
        image_path=image_path,
        language=language,
        checkpoint_label=checkpoint_label,
        threshold_value=mask_threshold,
        min_component_area=min_area,
    )
    renderer = OCRUI.__new__(OCRUI)
    print(renderer.render_cli_output(bundle))
    return 0


def parse_args() -> argparse.Namespace:
    parser = RussianArgumentParser(
        description="Интерфейс распознавания рукописного текста и анализа эмоций",
        add_help=False,
    )
    parser._optionals.title = "Параметры"
    parser.add_argument("-h", "--help", action="help", help="Показать это сообщение и выйти")
    parser.add_argument("--image", type=Path, help="Необязательный путь к изображению для запуска из командной строки")
    parser.add_argument("--language", choices=["ru", "en"], default="ru")
    parser.add_argument("--checkpoint", type=str, help="Относительный или абсолютный путь к .pth-файлу модели")
    parser.add_argument(
        "--threshold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Параметр совместимости: выбирает начальный режим предобработки в окне приложения",
    )
    parser.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD, help="Порог бинарной маски")
    parser.add_argument("--min-area", type=int, default=MIN_COMPONENT_AREA, help="Минимальная площадь связной компоненты")
    parser.add_argument("--no-gui", action="store_true", help="Запустить один анализ из командной строки без открытия окна")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_gui:
        if args.image is None:
            raise SystemExit("Для --no-gui требуется --image")
        return run_cli_analysis(args.image, args.language, args.checkpoint, args.mask_threshold, args.min_area)

    root = tk.Tk()
    app = OCRUI(root)
    if args.image is not None:
        app.file_var.set(str(args.image))
        app.language_var.set(args.language)
        app.refresh_checkpoint_options(force_default=True)
        if args.checkpoint:
            app.checkpoint_var.set(args.checkpoint)
        if args.threshold is not None:
            app.graph_variant_var.set("thresholded_segmented" if args.threshold else "baseline")
        app.mask_threshold_var.set(args.mask_threshold)
        app.min_area_var.set(args.min_area)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
