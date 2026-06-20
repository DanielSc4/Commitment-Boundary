"""Sequence assembly for full, no-CoT, and truncated early-exit runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from attributions.modeling import find_subsequence


@dataclass
class TraceContext:
    question_id: int
    trace_index: int
    gt_answer: str
    full_ids: List[int]
    prompt_len: int
    cot_start: int
    cot_end: int
    suffix_start: int
    n_start_think: int = 0
    n_end_think: int = 0
    task_id: Optional[str] = None
    entry_point: Optional[str] = None


def build_trace_context(
    question: dict,
    question_id: int,
    trace_index: int,
    tokenizer,
    end_ids: List[int],
    suffix_ids: List[int],
    start_ids: Optional[List[int]] = None,
    task_id: Optional[str] = None,
    entry_point: Optional[str] = None,
) -> Optional[TraceContext]:
    """Build ``prompt + complete CoT + answer-forcing suffix`` for one trace."""
    prompt_tokens = question["prompt_tokens"]
    traces_tokens = question["traces_tokens"]
    if trace_index >= len(traces_tokens):
        return None

    trace_tokens = traces_tokens[trace_index]
    end_position = find_subsequence(trace_tokens, end_ids)
    if end_position < 0:
        return None

    trace_through_end = trace_tokens[:end_position + len(end_ids)]
    full_ids = prompt_tokens + trace_through_end + suffix_ids
    prompt_len = len(prompt_tokens)
    cot_end = prompt_len + len(trace_through_end)
    n_start_think = (
        len(start_ids)
        if start_ids and trace_through_end[:len(start_ids)] == start_ids
        else 0
    )
    return TraceContext(
        question_id=question_id,
        trace_index=trace_index,
        gt_answer=str(question["GT_answer"]),
        full_ids=full_ids,
        prompt_len=prompt_len,
        cot_start=prompt_len,
        cot_end=cot_end,
        suffix_start=cot_end,
        n_start_think=n_start_think,
        n_end_think=len(end_ids),
        task_id=task_id,
        entry_point=entry_point,
    )
