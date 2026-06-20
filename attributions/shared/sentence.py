"""Sentence segmentation helpers for CoT traces.

The splitter works on decoded text but always returns token positions so later
stages can reuse the exact same sentence boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence


_ABBREVIATIONS = {
    "abbr",
    "approx",
    "cf",
    "dr",
    "e.g",
    "eg",
    "etc",
    "fig",
    "i.e",
    "ie",
    "mr",
    "mrs",
    "ms",
    "no",
    "prof",
    "sr",
    "st",
    "vs",
}

_LIST_MARKER_RE = re.compile(r"^\s*(?:\(?\d+|\(?[A-Za-z])\.$")
_WORD_BEFORE_DOT_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


@dataclass
class SentenceSpan:
    start_pos: int
    end_pos: int
    end_token_pos: int
    text: str


def decode_token_texts(tokenizer, token_ids: Sequence[int]) -> List[str]:
    """Decode each token independently without tokenizer cleanup."""
    return [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]


def build_token_char_spans(tokenizer, token_ids: Sequence[int]) -> tuple[str, List[str], List[int], List[int]]:
    """Return cumulative decoded text plus per-token char start/end offsets."""
    token_texts = decode_token_texts(tokenizer, token_ids)
    token_starts: List[int] = []
    token_ends: List[int] = []
    full_text_parts: List[str] = []
    cursor = 0

    for piece in token_texts:
        token_starts.append(cursor)
        full_text_parts.append(piece)
        cursor += len(piece)
        token_ends.append(cursor)

    return "".join(full_text_parts), token_texts, token_starts, token_ends


def _previous_visible_char(text: str, idx: int) -> str:
    j = idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return text[j] if j >= 0 else ""


def _next_visible_char(text: str, idx: int) -> str:
    j = idx + 1
    while j < len(text) and text[j].isspace():
        j += 1
    return text[j] if j < len(text) else ""


def _is_sentence_boundary(text: str, idx: int) -> bool:
    ch = text[idx]
    if ch not in ".!?":
        return False

    prev_visible = _previous_visible_char(text, idx)
    next_visible = _next_visible_char(text, idx)

    if ch == ".":
        if prev_visible.isdigit() and next_visible.isdigit():
            return False
        if (idx > 0 and text[idx - 1] == ".") or (idx + 1 < len(text) and text[idx + 1] == "."):
            return False

        prefix = text[max(0, idx - 32):idx + 1]
        line_prefix = text[text.rfind("\n", 0, idx) + 1:idx + 1]
        word_match = _WORD_BEFORE_DOT_RE.search(prefix)
        if word_match and word_match.group(1).casefold() in _ABBREVIATIONS:
            return False
        if _LIST_MARKER_RE.match(line_prefix):
            return False

    # Reject punctuation that is immediately followed by more punctuation,
    # except for closers like quotes or brackets.
    if idx + 1 < len(text) and text[idx + 1] in ":;,.":
        return False

    return True


def _extend_boundary_to_closers(text: str, idx: int) -> int:
    j = idx + 1
    while j < len(text) and text[j] in "\"')]}":
        j += 1
    return j


def _char_end_to_token_index(token_ends: Sequence[int], char_end: int) -> int:
    for token_idx, token_end in enumerate(token_ends):
        if token_end >= char_end:
            return token_idx
    return len(token_ends) - 1


def split_sentences_from_token_ids(tokenizer, token_ids: Sequence[int], offset: int = 0) -> List[SentenceSpan]:
    """Split token IDs into sentence spans, returning absolute token positions."""
    if not token_ids:
        return []

    text, _, token_starts, token_ends = build_token_char_spans(tokenizer, token_ids)
    spans: List[SentenceSpan] = []
    sentence_start_token = 0

    for char_idx, ch in enumerate(text):
        if not _is_sentence_boundary(text, char_idx):
            continue

        char_end = _extend_boundary_to_closers(text, char_idx)
        end_token_idx = _char_end_to_token_index(token_ends, char_end)
        if end_token_idx < sentence_start_token:
            continue

        sent_text = text[token_starts[sentence_start_token]:token_ends[end_token_idx]]
        if sent_text.strip():
            spans.append(
                SentenceSpan(
                    start_pos=offset + sentence_start_token,
                    end_pos=offset + end_token_idx + 1,
                    end_token_pos=offset + end_token_idx,
                    text=sent_text,
                )
            )
        sentence_start_token = end_token_idx + 1

    if sentence_start_token < len(token_ids):
        trailing_text = text[token_starts[sentence_start_token]:]
        if trailing_text.strip():
            spans.append(
                SentenceSpan(
                    start_pos=offset + sentence_start_token,
                    end_pos=offset + len(token_ids),
                    end_token_pos=offset + len(token_ids) - 1,
                    text=trailing_text,
                )
            )

    return spans


def expand_sentence_spans(spans: Sequence[SentenceSpan | dict]) -> List[int]:
    """Expand sentence spans into sorted token positions."""
    positions: List[int] = []
    for span in spans:
        start = span.start_pos if isinstance(span, SentenceSpan) else span["start_pos"]
        end = span.end_pos if isinstance(span, SentenceSpan) else span["end_pos"]
        positions.extend(range(start, end))
    return sorted(set(positions))
