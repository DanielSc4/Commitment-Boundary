#!/usr/bin/env python3
"""vLLM port of sentence-level causal CoT attribution.

Produces output identical in schema (and ~identical numerically) to
``attributions/nnsight_sentence_causal.py``, but batches all prefix passes for
a trace through vLLM.

For each trace with N CoT sentences, builds N+1 prefixes::

    k=0 : [prompt][start_cot][end_cot][suffix]                      (no-CoT)
    k=1 : [prompt][start_cot][sent_1][end_cot][suffix]
    ...
    k=N : [prompt][start_cot][sent_1 ... sent_N][end_cot][suffix]   (full CoT)

The target token id ``full_token_id`` is taken as the argmax of the first
generated token after prefix k=N. We then read ``P(full_token_id | prefix_k)``
for every k via vLLM's ``prompt_logprobs`` by appending ``full_token_id`` to
each sequence -- one batched forward pass per trace.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from attributions.utils import (
    extract_boxed_answer,
    get_answer_domain,
    get_answer_suffix,
    get_reasoning_traces,
    get_thinking_tokens,
    is_code_benchmark,
)
from attributions.shared import split_sentences_from_token_ids


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return the start index of ``needle`` in ``haystack`` or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def _resolve_marker_ids(tokenizer, marker_text: str, marker_ids: Optional[List[int]]) -> List[int]:
    if marker_ids is not None:
        return marker_ids
    return tokenizer.encode(marker_text, add_special_tokens=False)


def _build_prefix_ids(
    prompt_tokens: List[int],
    start_ids: List[int],
    reasoning_tokens: List[int],
    prefix_end: int,
    end_ids: List[int],
    suffix_ids: List[int],
) -> List[int]:
    return prompt_tokens + start_ids + reasoning_tokens[:prefix_end] + end_ids + suffix_ids


SEMANTIC_GUESS_LABELS = {
    0: "no_guess",
    1: "wrong_guess",
    2: "final_equivalent_guess",
}


def _semantic_label_name(label: int) -> str:
    return SEMANTIC_GUESS_LABELS.get(int(label), f"unknown_{label}")


def _normalize_answer_text(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""

    text = text.strip("$").strip()
    text = re.sub(r"^\\text\{(.+)\}$", r"\1", text).strip()
    if re.fullmatch(r"\(?[A-Za-z]\)?\.?", text):
        text = text.strip("().").upper()
    return re.sub(r"\s+", "", text).casefold()


def _normalize_mcq_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""
    text = text.strip("$").strip()
    text = re.sub(r"^\\text\{(.+)\}$", r"\1", text).strip()
    match = re.match(r"^\(?\s*([A-Za-z])\s*\)?\s*(?:[.:)\]]|\s|$)", text)
    if match:
        return match.group(1).casefold()
    return _normalize_answer_text(text)


def _extract_answer_from_continuation(answer_suffix: str, generated: str) -> str:
    if not generated:
        return ""
    if r"\boxed{" in answer_suffix:
        return extract_boxed_answer(answer_suffix + generated).strip()
    return generated.strip()


def _math_parse_variants(answer: str) -> list[str]:
    text = str(answer).strip()
    if not text:
        return []

    normalized = (
        text
        .replace(r"\tfrac", r"\frac")
        .replace(r"\dfrac", r"\frac")
        .replace(r"\left", "")
        .replace(r"\right", "")
    )
    variants = [text, f"${text}$", normalized, f"${normalized}$"]

    seen = set()
    unique = []
    for variant in variants:
        if variant and variant not in seen:
            unique.append(variant)
            seen.add(variant)
    return unique


def _math_verify_with_variants(reference: str, candidate: str) -> dict[str, Any]:
    try:
        from math_verify import parse, verify

        last_error = None
        for ref_variant in _math_parse_variants(reference):
            try:
                reference_parsed = parse(ref_variant)
            except Exception as exc:  # noqa: BLE001 - try the next variant
                last_error = str(exc)
                continue
            if not reference_parsed:
                continue

            for candidate_variant in _math_parse_variants(candidate):
                try:
                    candidate_parsed = parse(candidate_variant)
                except Exception as exc:  # noqa: BLE001 - try the next variant
                    last_error = str(exc)
                    continue
                if candidate_parsed and verify(reference_parsed, candidate_parsed):
                    return {
                        "equivalent": True,
                        "method": "math_verify",
                        "error": None,
                        "reference_variant": ref_variant,
                        "candidate_variant": candidate_variant,
                    }

        return {
            "equivalent": False,
            "method": "math_verify",
            "error": last_error,
            "reference_variant": None,
            "candidate_variant": None,
        }
    except Exception as exc:  # noqa: BLE001 - verifier failures should not stop attribution
        return {
            "equivalent": False,
            "method": "verification_error",
            "error": str(exc),
            "reference_variant": None,
            "candidate_variant": None,
        }


def _verify_answer_equivalence(reference: str, candidate: str, answer_domain: str = "math") -> dict[str, Any]:
    if answer_domain == "mcq":
        ref_norm = _normalize_mcq_answer(reference)
        cand_norm = _normalize_mcq_answer(candidate)
        if not cand_norm:
            return {"equivalent": False, "method": "no_guess", "error": None}
        if not ref_norm:
            return {"equivalent": False, "method": "missing_reference", "error": None}
        return {"equivalent": cand_norm == ref_norm, "method": "mcq_exact", "error": None}

    if answer_domain == "string":
        ref_norm = _normalize_answer_text(reference)
        cand_norm = _normalize_answer_text(candidate)
        if not cand_norm:
            return {"equivalent": False, "method": "no_guess", "error": None}
        if not ref_norm:
            return {"equivalent": False, "method": "missing_reference", "error": None}
        return {"equivalent": cand_norm == ref_norm, "method": "normalized_exact", "error": None}

    ref_norm = _normalize_answer_text(reference)
    cand_norm = _normalize_answer_text(candidate)
    if not cand_norm:
        return {"equivalent": False, "method": "no_guess", "error": None}
    if not ref_norm:
        return {"equivalent": False, "method": "missing_reference", "error": None}
    if cand_norm == ref_norm:
        return {"equivalent": True, "method": "normalized_exact", "error": None}
    return _math_verify_with_variants(reference, candidate)


def _make_semantic_guess_info(
    tokenizer,
    answer_suffix: str,
    generated_ids: list[int],
    generated_continuation: str,
    reference_answer: str,
    source: str,
    answer_domain: str = "math",
    sentence_index: Optional[int] = None,
) -> dict[str, Any]:
    extracted_answer = _extract_answer_from_continuation(answer_suffix, generated_continuation)
    equivalence = _verify_answer_equivalence(reference_answer, extracted_answer, answer_domain)
    if not _normalize_answer_text(extracted_answer):
        label = 0
    elif equivalence["equivalent"]:
        label = 2
    else:
        label = 1

    first_token_id = int(generated_ids[0]) if generated_ids else None
    return {
        "source": source,
        "sentence_index": sentence_index,
        "generated_continuation": generated_continuation,
        "extracted_answer": extracted_answer,
        "normalized_answer": _normalize_answer_text(extracted_answer),
        "answer_domain": answer_domain,
        "semantic_label": label,
        "semantic_label_name": _semantic_label_name(label),
        "is_final_equivalent": bool(label == 2),
        "equivalence_method": equivalence["method"],
        "equivalence_error": equivalence["error"],
        "first_token_id": first_token_id,
        "first_token": tokenizer.decode([first_token_id]) if first_token_id is not None else None,
    }


def _apply_clue_alpha_gate(
    info: dict[str, Any],
    first_token_prob: float,
    threshold: float,
    clue_alpha: float,
) -> dict[str, Any]:
    raw_label = int(info["semantic_label"])
    below_threshold = float(first_token_prob) < float(threshold)
    info.update(
        {
            "raw_semantic_label": raw_label,
            "raw_semantic_label_name": _semantic_label_name(raw_label),
            "clue_alpha": float(clue_alpha),
            "clue_alpha_threshold": float(threshold),
            "clue_alpha_first_token_prob": float(first_token_prob),
            "below_clue_alpha_threshold": bool(below_threshold),
            "clue_alpha_overridden": bool(below_threshold and raw_label != 0),
        }
    )
    if below_threshold:
        info["semantic_label"] = 0
        info["semantic_label_name"] = _semantic_label_name(0)
        info["is_final_equivalent"] = False
    return info


def _semantic_candidate_summary(guess_infos: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, Optional[int], int], dict[str, Any]] = {}
    for info in guess_infos:
        key = (info.get("normalized_answer") or "", info.get("first_token_id"), int(info["semantic_label"]))
        if key not in grouped:
            grouped[key] = {
                "extracted_answer": info.get("extracted_answer", ""),
                "normalized_answer": info.get("normalized_answer", ""),
                "semantic_label": int(info["semantic_label"]),
                "semantic_label_name": info.get("semantic_label_name"),
                "is_final_equivalent": bool(info.get("is_final_equivalent")),
                "first_token_id": info.get("first_token_id"),
                "first_token": info.get("first_token"),
                "sources": [],
                "generated_continuations": [],
            }
        source = {"source": info.get("source")}
        if info.get("sentence_index") is not None:
            source["sentence_index"] = int(info["sentence_index"])
        grouped[key]["sources"].append(source)
        continuation = info.get("generated_continuation")
        if continuation not in grouped[key]["generated_continuations"]:
            grouped[key]["generated_continuations"].append(continuation)

    candidates = list(grouped.values())
    candidates.sort(key=lambda c: (c["semantic_label"], str(c["first_token_id"]), c["normalized_answer"]))

    by_token: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        token_id = candidate.get("first_token_id")
        if token_id is None:
            continue
        by_token.setdefault(int(token_id), []).append(candidate)

    collisions = []
    for token_id, token_candidates in sorted(by_token.items()):
        distinct_answers = {
            c.get("normalized_answer") or "|".join(c.get("generated_continuations") or [])
            for c in token_candidates
        }
        labels = sorted({int(c["semantic_label"]) for c in token_candidates})
        if len(distinct_answers) <= 1:
            continue
        collisions.append(
            {
                "first_token_id": token_id,
                "first_token": token_candidates[0].get("first_token"),
                "semantic_labels": labels,
                "cross_class": len(labels) > 1,
                "candidates": [
                    {
                        "extracted_answer": c.get("extracted_answer"),
                        "semantic_label": c.get("semantic_label"),
                        "semantic_label_name": c.get("semantic_label_name"),
                        "sources": c.get("sources", []),
                    }
                    for c in token_candidates
                ],
            }
        )

    def _ids_for(label: int) -> list[int]:
        return sorted({
            int(c["first_token_id"])
            for c in candidates
            if c.get("first_token_id") is not None and int(c["semantic_label"]) == label
        })

    final_ids = _ids_for(2)
    wrong_ids = _ids_for(1)
    all_guess_ids = sorted(set(final_ids) | set(wrong_ids))
    return {
        "discovered_candidates": candidates,
        "first_token_collisions": collisions,
        "final_equivalent_first_token_ids": final_ids,
        "wrong_guess_first_token_ids": wrong_ids,
        "all_guess_first_token_ids": all_guess_ids,
    }


def _topk_prob_mass(topk_token_probs: dict[int, float], token_ids: list[int]) -> tuple[float, list[int]]:
    unique_ids = sorted(set(int(t) for t in token_ids))
    mass = 0.0
    missing = []
    for token_id in unique_ids:
        if token_id in topk_token_probs:
            mass += float(topk_token_probs[token_id])
        else:
            missing.append(token_id)
    return mass, missing


def _decode_generated(tokenizer, token_ids: list[int], max_tokens: Optional[int] = None) -> str:
    limited = token_ids if max_tokens is None else token_ids[:max_tokens]
    return tokenizer.decode(limited, skip_special_tokens=True).strip()


def _prefix_eval_from_vllm_output(out, tokenizer) -> dict[str, Any]:
    completion = out.outputs[0]
    token_ids = [int(token_id) for token_id in completion.token_ids]
    first_token_id = token_ids[0] if token_ids else None
    first_logprobs = completion.logprobs[0] if completion.logprobs and token_ids else {}
    first_token_prob = 0.0
    if first_token_id is not None and first_token_id in first_logprobs:
        first_token_prob = float(math.exp(first_logprobs[first_token_id].logprob))
    topk_token_probs = {
        int(token_id): float(math.exp(logprob.logprob))
        for token_id, logprob in first_logprobs.items()
    }
    return {
        "generated_ids": token_ids,
        "generated_text": _decode_generated(tokenizer, token_ids),
        "first_token_id": first_token_id,
        "first_token": tokenizer.decode([first_token_id]) if first_token_id is not None else None,
        "first_token_prob": first_token_prob,
        "topk_token_probs": topk_token_probs,
    }


def _hf_model_block(llm: LLM, model_name: str) -> dict:
    hf_cfg = llm.llm_engine.model_config.hf_config
    n_heads = getattr(hf_cfg, "num_attention_heads", None)
    n_kv = getattr(hf_cfg, "num_key_value_heads", n_heads)
    return {
        "name": model_name,
        "n_layers": getattr(hf_cfg, "num_hidden_layers", None),
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "d_model": getattr(hf_cfg, "hidden_size", None),
    }


def _gen_first_token(out, tokenizer) -> tuple[int, float, str]:
    """Extract (first_token_id, first_token_prob, decoded_response) from one gen output."""
    completion = out.outputs[0]
    token_ids = list(completion.token_ids)
    first_id = int(token_ids[0])
    first_lp = completion.logprobs[0][first_id].logprob
    first_prob = float(math.exp(first_lp))
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return first_id, first_prob, text


def _prob_of_target(out, target_id: int) -> float:
    """Read P(target_id) from the final prompt position of a prompt_logprobs output."""
    last = out.prompt_logprobs[-1]
    if last is None or target_id not in last:
        return 0.0
    return float(math.exp(last[target_id].logprob))


def _argmax_at_last(out) -> tuple[Optional[int], float]:
    """Return (argmax_token_id, prob) at the final prompt position."""
    last = out.prompt_logprobs[-1]
    if not last:
        return None, 0.0
    best_id, best_lp = None, -float("inf")
    for tid, lp in last.items():
        lp_val = lp.logprob
        if lp_val > best_lp:
            best_lp = lp_val
            best_id = int(tid)
    if best_id is None:
        return None, 0.0
    return best_id, float(math.exp(best_lp))


def _reset_prefix_cache(llm: LLM) -> None:
    """Best-effort reset of vLLM's prefix cache between independent traces."""
    reset = getattr(llm, "reset_prefix_cache", None)
    if reset is None:
        return
    try:
        reset()
    except Exception as exc:
        tqdm.write(f"  WARNING: failed to reset vLLM prefix cache: {exc}")


def _gib(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def _model_vocab_size(llm: LLM, tokenizer) -> int:
    hf_cfg = llm.llm_engine.model_config.hf_config
    return int(getattr(hf_cfg, "vocab_size", len(tokenizer)))


def _free_cuda_memory_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes)


def _estimate_prompt_logprobs_bytes(
    prefix_ids_list: Sequence[Sequence[int]],
    vocab_size: int,
    prefill_step_tokens: int,
    max_prompt_logprob_chunk_positions: int,
    forced_batch_size: int,
) -> int:
    """Estimate vLLM's largest FP32 full-vocab prompt_logprobs log_softmax allocation."""
    if not prefix_ids_list:
        return 0
    max_scheduled_positions = 0
    for chunk_start, chunk_end in _prompt_logprob_chunk_spans(
        prefix_ids_list,
        max_prompt_logprob_chunk_positions,
        forced_batch_size,
    ):
        chunk = prefix_ids_list[chunk_start:chunk_end]
        # We append the target token before calling prompt_logprobs. vLLM computes
        # prompt logprobs for all prompt positions after the first token, so each
        # forced prompt contributes len(prefix_ids) positions. The scheduler may
        # pack several requests into one 8192-token prefill step.
        chunk_positions = sum(len(prefix_ids) for prefix_ids in chunk)
        max_scheduled_positions = max(
            max_scheduled_positions,
            min(chunk_positions, prefill_step_tokens),
        )
    return int(max_scheduled_positions * vocab_size * 4)


def _prompt_logprob_chunk_spans(
    prefix_ids_list: Sequence[Sequence[int]],
    max_prompt_logprob_chunk_positions: int,
    max_batch_size: int,
) -> list[tuple[int, int]]:
    """Chunk prefix indices by prompt positions, with batch size as a secondary cap."""
    if max_prompt_logprob_chunk_positions <= 0:
        raise ValueError("max_prompt_logprob_chunk_positions must be positive")
    if max_batch_size <= 0:
        raise ValueError("forced_batch_size must be positive")

    spans: list[tuple[int, int]] = []
    chunk_start = 0
    chunk_positions = 0
    for idx, prefix_ids in enumerate(prefix_ids_list):
        prefix_len = len(prefix_ids)
        chunk_size = idx - chunk_start
        should_flush = chunk_size > 0 and (
            chunk_positions + prefix_len > max_prompt_logprob_chunk_positions
            or chunk_size >= max_batch_size
        )
        if should_flush:
            spans.append((chunk_start, idx))
            chunk_start = idx
            chunk_positions = 0
        chunk_positions += prefix_len

    if chunk_start < len(prefix_ids_list):
        spans.append((chunk_start, len(prefix_ids_list)))
    return spans


def _should_skip_for_prompt_logprobs_oom(
    prefix_ids_list: Sequence[Sequence[int]],
    vocab_size: int,
    prefill_step_tokens: int,
    safety_margin_gib: float,
    max_prompt_logprob_chunk_positions: int,
    forced_batch_size: int,
) -> tuple[bool, dict]:
    estimated_bytes = _estimate_prompt_logprobs_bytes(
        prefix_ids_list,
        vocab_size,
        prefill_step_tokens,
        max_prompt_logprob_chunk_positions,
        forced_batch_size,
    )
    free_bytes = _free_cuda_memory_bytes()
    margin_bytes = int(safety_margin_gib * 1024 ** 3)
    available_bytes = None if free_bytes is None else max(0, free_bytes - margin_bytes)
    should_skip = available_bytes is not None and estimated_bytes > available_bytes
    return should_skip, {
        "estimated_prompt_logprobs_allocation_gib": _gib(estimated_bytes),
        "cuda_free_gib": None if free_bytes is None else _gib(free_bytes),
        "cuda_available_after_margin_gib": None if available_bytes is None else _gib(available_bytes),
        "oom_preflight_safety_margin_gib": safety_margin_gib,
        "prompt_logprobs_prefill_step_tokens": prefill_step_tokens,
        "max_prompt_logprob_chunk_positions": max_prompt_logprob_chunk_positions,
        "forced_batch_size": forced_batch_size,
        "vocab_size": vocab_size,
    }


def main(
    model: str,
    data_name: str,
    question_ids: Optional[Sequence[int]] = None,
    trace_indices: Optional[Sequence[int]] = None,
    max_questions: Optional[int] = None,
    output_dir: Optional[str] = None,
    comparison_max_new_tokens: int = 64,
    semantic_guess_labels: bool = True,
    semantic_max_new_tokens: Optional[int] = None,
    semantic_first_token_topk: int = 50,
    clue_alpha: float = 0.5,
    answer_domain: str = "auto",
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    seed: int = 0,
    forced_batch_size: int = 32,
    overwrite: bool = False,
    enable_prefix_caching: bool = True,
    reset_prefix_cache_per_trace: bool = True,
    oom_preflight: bool = True,
    oom_preflight_before_generation: bool = False,
    oom_preflight_safety_margin_gib: float = 0.50,
    prompt_logprobs_prefill_step_tokens: int = 8192,
    max_prompt_logprob_chunk_positions: int = 4096,
):
    """Compute sentence-level causal importance scores via vLLM."""
    if forced_batch_size <= 0:
        raise ValueError("forced_batch_size must be positive")
    if prompt_logprobs_prefill_step_tokens <= 0:
        raise ValueError("prompt_logprobs_prefill_step_tokens must be positive")
    if max_prompt_logprob_chunk_positions <= 0:
        raise ValueError("max_prompt_logprob_chunk_positions must be positive")

    print(f"Loading model (vLLM): {model}")
    requested_max_logprobs = max(20, int(semantic_first_token_topk) if semantic_guess_labels else 1)
    llm_kwargs = dict(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        max_logprobs=requested_max_logprobs,
        seed=seed,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if "ministral" in model.lower() or "mistral" in model.lower():
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    thinking_tokens = get_thinking_tokens(model)
    start_ids = _resolve_marker_ids(tokenizer, thinking_tokens["start_token"], thinking_tokens.get("start_token_ids"))
    end_ids = _resolve_marker_ids(tokenizer, thinking_tokens["end_token"], thinking_tokens.get("end_token_ids"))
    answer_suffix = get_answer_suffix(data_name)
    suffix_ids = tokenizer.encode(answer_suffix, add_special_tokens=False)

    if is_code_benchmark(data_name) and comparison_max_new_tokens == 64:
        comparison_max_new_tokens = 256
    resolved_answer_domain = get_answer_domain(data_name) if answer_domain == "auto" else answer_domain
    if resolved_answer_domain not in {"math", "mcq", "string", "code"}:
        raise ValueError(
            f"Unknown answer_domain={resolved_answer_domain!r}; expected auto, math, mcq, string, or code"
        )
    semantic_max_new_tokens = (
        comparison_max_new_tokens if semantic_max_new_tokens is None else semantic_max_new_tokens
    )
    semantic_enabled = bool(semantic_guess_labels) and resolved_answer_domain != "code"
    if semantic_guess_labels and not semantic_enabled:
        print("Semantic guess labels: disabled for code benchmarks")
    else:
        print(
            "Semantic guess labels: "
            f"{'enabled' if semantic_enabled else 'disabled'}"
            + (f" (max_new_tokens={semantic_max_new_tokens})" if semantic_enabled else "")
        )
    if semantic_enabled:
        print(f"Semantic clue gate: clue_alpha={float(clue_alpha):.4g}")
    print(f"Answer domain: {resolved_answer_domain}")

    traces_data = get_reasoning_traces(model, data_name)
    print(f"Loaded {len(traces_data)} questions from traces")

    model_short = model.split("/")[-1]
    data_short = data_name.split("/")[-1]
    out_dir = (
        Path(output_dir)
        if output_dir
        else Path("outputs") / model_short / data_short / "contribution_graphs" / "sentence_causal" / "boxed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    question_ids = list(question_ids) if question_ids is not None else list(range(len(traces_data)))
    if max_questions is not None:
        question_ids = question_ids[:max_questions]

    generation_max_new_tokens = (
        max(comparison_max_new_tokens, semantic_max_new_tokens)
        if semantic_enabled
        else comparison_max_new_tokens
    )
    continuation_logprobs = max(1, int(semantic_first_token_topk) if semantic_enabled else 1)
    gen_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=generation_max_new_tokens,
        logprobs=continuation_logprobs,
        skip_special_tokens=False,
        seed=seed,
    )
    forced_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=1,
        skip_special_tokens=False,
        seed=seed,
    )

    model_block = _hf_model_block(llm, model)
    vocab_size = _model_vocab_size(llm, tokenizer)
    print(
        f"Model: {model_block['n_layers']} layers, {model_block['n_heads']} heads, "
        f"d_model={model_block['d_model']}, vocab_size={vocab_size}"
    )
    print(
        "Prompt-logprob chunking: "
        f"max_positions={max_prompt_logprob_chunk_positions}, "
        f"max_prefixes={forced_batch_size}"
    )
    if oom_preflight:
        print(
            "OOM preflight enabled: "
            f"prefill_step_tokens={prompt_logprobs_prefill_step_tokens}, "
            f"max_chunk_positions={max_prompt_logprob_chunk_positions}, "
            f"safety_margin={oom_preflight_safety_margin_gib:.2f} GiB"
        )

    for q_id in question_ids:
        tqdm.write("[----------]")
        if q_id >= len(traces_data):
            tqdm.write(f"WARNING: question_id {q_id} out of range, skipping")
            continue

        output_file = out_dir / f"question_{q_id:04d}.json"
        if output_file.exists() and not overwrite:
            tqdm.write(f"  Q{q_id}: {output_file.name} exists, skipping (use --overwrite to force)")
            continue

        q_data = traces_data[q_id]
        tqdm.write(f"Q{q_id} lens: " + ", ".join(
            f"t{t_idx}={len(q_data['traces_tokens'][t_idx])}" for t_idx in range(len(traces_data[q_id]["traces_tokens"]))
        ))

        prompt_tokens: List[int] = q_data["prompt_tokens"]
        traces_tokens: List[List[int]] = q_data["traces_tokens"]
        input_text = q_data["input_text"]
        gt_answer = q_data["GT_answer"]
        extracted_answers = q_data.get("extracted_answers", [])

        trace_results = []
        active_trace_indices = (
            list(trace_indices) if trace_indices is not None else list(range(len(traces_tokens)))
        )

        for t_idx in tqdm(active_trace_indices, desc=f"Q{q_id} traces", unit="trace", dynamic_ncols=True):
            if t_idx >= len(traces_tokens):
                tqdm.write(f"  Q{q_id} trace {t_idx}: out of range, skipping")
                continue

            trace_tokens = traces_tokens[t_idx]
            end_pos = _find_subsequence(trace_tokens, end_ids)
            if end_pos == -1:
                tqdm.write(f"  Q{q_id} trace {t_idx}: no end-thinking marker found, skipping")
                continue

            has_start_marker = bool(start_ids) and trace_tokens[:len(start_ids)] == start_ids
            trace_start_ids = start_ids if has_start_marker else []
            reasoning_start = len(trace_start_ids)
            reasoning_tokens = trace_tokens[reasoning_start:end_pos]
            full_trace_ids = prompt_tokens + trace_tokens[:end_pos + len(end_ids)] + suffix_ids

            sentence_spans = split_sentences_from_token_ids(
                tokenizer,
                reasoning_tokens,
                offset=len(prompt_tokens) + reasoning_start,
            )
            if not sentence_spans:
                tqdm.write(f"  Q{q_id} trace {t_idx}: no non-empty CoT sentences found, skipping")
                continue

            no_cot_ids = prompt_tokens + trace_start_ids + end_ids + suffix_ids

            prefix_ids_list: List[List[int]] = [no_cot_ids]
            for span in sentence_spans:
                prefix_end = span.end_pos - (len(prompt_tokens) + reasoning_start)
                prefix_ids_list.append(
                    _build_prefix_ids(
                        prompt_tokens,
                        trace_start_ids,
                        reasoning_tokens,
                        prefix_end,
                        end_ids,
                        suffix_ids,
                    )
                )

            if oom_preflight and oom_preflight_before_generation:
                skip_for_oom, oom_info = _should_skip_for_prompt_logprobs_oom(
                    prefix_ids_list,
                    vocab_size,
                    prompt_logprobs_prefill_step_tokens,
                    oom_preflight_safety_margin_gib,
                    max_prompt_logprob_chunk_positions,
                    forced_batch_size,
                )
                if skip_for_oom:
                    trace_results.append(
                        {
                            "trace_index": t_idx,
                            "skipped": True,
                            "skip_reason": "oom_preflight_prompt_logprobs",
                            "n_tokens": len(full_trace_ids),
                            "n_sentences": len(sentence_spans),
                            **oom_info,
                        }
                    )
                    tqdm.write(
                        f"  [x] Q{q_id} trace {t_idx}: OOM preflight skipped "
                        f"({oom_info['estimated_prompt_logprobs_allocation_gib']:.2f} GiB needed, "
                        f"{oom_info['cuda_available_after_margin_gib']:.2f} GiB available after margin)"
                    )
                    continue

            try:
                t0 = time.time()

                prefix_evals: Optional[list[dict[str, Any]]] = None
                if semantic_enabled:
                    prefix_evals = []
                    continuation_prompts = [
                        TokensPrompt(prompt_token_ids=pids) for pids in prefix_ids_list
                    ]
                    for chunk_start in range(0, len(continuation_prompts), forced_batch_size):
                        chunk = continuation_prompts[chunk_start:chunk_start + forced_batch_size]
                        chunk_outs = llm.generate(chunk, gen_params, use_tqdm=False)
                        prefix_evals.extend(
                            _prefix_eval_from_vllm_output(out, tokenizer)
                            for out in chunk_outs
                        )
                    no_cot_eval = prefix_evals[0]
                    full_eval = prefix_evals[-1]
                    full_token_id = full_eval["first_token_id"]
                    full_token_prob = full_eval["first_token_prob"]
                    full_response = _decode_generated(
                        tokenizer,
                        full_eval["generated_ids"],
                        comparison_max_new_tokens,
                    )
                    no_cot_response = _decode_generated(
                        tokenizer,
                        no_cot_eval["generated_ids"],
                        comparison_max_new_tokens,
                    )
                    if full_token_id is None:
                        trace_results.append(
                            {
                                "trace_index": t_idx,
                                "skipped": True,
                                "skip_reason": "full_cot_generated_no_tokens",
                            }
                        )
                        tqdm.write(f"  Q{q_id} trace {t_idx}: full-CoT generated no tokens, skipped")
                        continue
                else:
                    gen_outs = llm.generate(
                        [
                            TokensPrompt(prompt_token_ids=full_trace_ids),
                            TokensPrompt(prompt_token_ids=no_cot_ids),
                        ],
                        gen_params,
                        use_tqdm=False,
                    )
                    full_token_id, full_token_prob, full_response = _gen_first_token(gen_outs[0], tokenizer)
                    _, _, no_cot_response = _gen_first_token(gen_outs[1], tokenizer)

                if full_response == no_cot_response:
                    trace_results.append(
                        {
                            "trace_index": t_idx,
                            "skipped": True,
                            "skip_reason": "full_response_matches_no_cot",
                            "full_response": full_response,
                            "no_cot_response": no_cot_response,
                        }
                    )
                    tqdm.write(f"  Q{q_id} trace {t_idx}: full/no-CoT response identical, skipped")
                    continue

                no_cot_force_out = llm.generate(
                    [TokensPrompt(prompt_token_ids=no_cot_ids + [full_token_id])],
                    forced_params,
                    use_tqdm=False,
                )
                no_cot_prob_early = _prob_of_target(no_cot_force_out[0], full_token_id)
                if (
                    full_token_prob <= no_cot_prob_early
                    or (full_token_prob - no_cot_prob_early) <= 0.5
                ):
                    trace_results.append(
                        {
                            "trace_index": t_idx,
                            "skipped": True,
                            "skip_reason": "full_cot_does_not_bring_benefit",
                            "target_token_id": full_token_id,
                            "target_token": tokenizer.decode([full_token_id]),
                            "target_token_prob_full_cot": full_token_prob,
                            "target_token_prob_no_cot": no_cot_prob_early,
                            "full_response": full_response,
                            "no_cot_response": no_cot_response,
                        }
                    )
                    tqdm.write(
                        f"  Q{q_id} trace {t_idx}: full CoT does not bring benefit "
                        f"(p_full={full_token_prob:.3f}, p_no_cot={no_cot_prob_early:.3f}), skipped"
                    )
                    continue

                if oom_preflight:
                    skip_for_oom, oom_info = _should_skip_for_prompt_logprobs_oom(
                        prefix_ids_list,
                        vocab_size,
                        prompt_logprobs_prefill_step_tokens,
                        oom_preflight_safety_margin_gib,
                        max_prompt_logprob_chunk_positions,
                        forced_batch_size,
                    )
                    if skip_for_oom:
                        trace_results.append(
                            {
                                "trace_index": t_idx,
                                "skipped": True,
                                "skip_reason": "oom_preflight_prompt_logprobs",
                                "n_tokens": len(full_trace_ids),
                                "n_sentences": len(sentence_spans),
                                "target_token_id": full_token_id,
                                "target_token": tokenizer.decode([full_token_id]),
                                "target_token_prob_full_cot": full_token_prob,
                                "full_response": full_response,
                                "no_cot_response": no_cot_response,
                                **oom_info,
                            }
                        )
                        tqdm.write(
                            f"  [x] Q{q_id} trace {t_idx}: OOM preflight skipped "
                            f"({oom_info['estimated_prompt_logprobs_allocation_gib']:.2f} GiB needed, "
                            f"{oom_info['cuda_available_after_margin_gib']:.2f} GiB available after margin)"
                        )
                        continue

                forced_batch = [
                    TokensPrompt(prompt_token_ids=pids + [full_token_id]) for pids in prefix_ids_list
                ]
                prefix_probs: List[float] = []
                prefix_pred_tokens: List[tuple[Optional[int], float]] = []
                mid_trace_oom_skip: Optional[dict] = None
                for chunk_start, chunk_end in _prompt_logprob_chunk_spans(
                    prefix_ids_list,
                    max_prompt_logprob_chunk_positions,
                    forced_batch_size,
                ):
                    chunk = forced_batch[chunk_start:chunk_end]
                    if oom_preflight:
                        chunk_prefix_ids = prefix_ids_list[chunk_start:chunk_end]
                        skip_for_oom, oom_info = _should_skip_for_prompt_logprobs_oom(
                            chunk_prefix_ids,
                            vocab_size,
                            prompt_logprobs_prefill_step_tokens,
                            oom_preflight_safety_margin_gib,
                            max_prompt_logprob_chunk_positions,
                            len(chunk),
                        )
                        if skip_for_oom:
                            mid_trace_oom_skip = {
                                "trace_index": t_idx,
                                "skipped": True,
                                "skip_reason": "oom_preflight_prompt_logprobs",
                                "n_tokens": len(full_trace_ids),
                                "n_sentences": len(sentence_spans),
                                "target_token_id": full_token_id,
                                "target_token": tokenizer.decode([full_token_id]),
                                "target_token_prob_full_cot": full_token_prob,
                                "full_response": full_response,
                                "no_cot_response": no_cot_response,
                                "skipped_at_forced_chunk_start": chunk_start,
                                **oom_info,
                            }
                            tqdm.write(
                                f"  [x] Q{q_id} trace {t_idx}: OOM preflight skipped at forced chunk "
                                f"{chunk_start}-{chunk_end} "
                                f"({oom_info['estimated_prompt_logprobs_allocation_gib']:.2f} GiB needed, "
                                f"{oom_info['cuda_available_after_margin_gib']:.2f} GiB available after margin)"
                            )
                            break
                    chunk_outs = llm.generate(chunk, forced_params, use_tqdm=False)
                    prefix_probs.extend(_prob_of_target(o, full_token_id) for o in chunk_outs)
                    prefix_pred_tokens.extend(_argmax_at_last(o) for o in chunk_outs)
                if mid_trace_oom_skip is not None:
                    trace_results.append(mid_trace_oom_skip)
                    continue
                no_cot_prob = prefix_probs[0]

                semantic_sentence_infos: list[dict[str, Any]] = []
                semantic_trace_meta: Optional[dict[str, Any]] = None
                semantic_prefix_final_masses: list[float] = []
                semantic_prefix_wrong_masses: list[float] = []
                semantic_prefix_all_masses: list[float] = []
                semantic_missing_final_by_prefix: list[dict[str, Any]] = []
                semantic_missing_wrong_by_prefix: list[dict[str, Any]] = []
                semantic_missing_all_by_prefix: list[dict[str, Any]] = []
                if semantic_enabled:
                    if prefix_evals is None:
                        raise RuntimeError("Internal error: semantic labels enabled without prefix evaluations")

                    semantic_clue_threshold = float(no_cot_prob) + float(clue_alpha) * (
                        float(full_token_prob) - float(no_cot_prob)
                    )
                    full_semantic_gen_ids = prefix_evals[-1]["generated_ids"][:semantic_max_new_tokens]
                    full_semantic_response = _decode_generated(tokenizer, full_semantic_gen_ids)
                    full_semantic_answer = _extract_answer_from_continuation(
                        answer_suffix,
                        full_semantic_response,
                    )
                    full_semantic_info = _make_semantic_guess_info(
                        tokenizer,
                        answer_suffix,
                        full_semantic_gen_ids,
                        full_semantic_response,
                        full_semantic_answer,
                        source="full_cot",
                        answer_domain=resolved_answer_domain,
                    )

                    for sent_idx, prefix_eval in enumerate(prefix_evals[1:]):
                        sentence_gen_ids = prefix_eval["generated_ids"][:semantic_max_new_tokens]
                        sentence_response = _decode_generated(tokenizer, sentence_gen_ids)
                        semantic_info = _make_semantic_guess_info(
                            tokenizer,
                            answer_suffix,
                            sentence_gen_ids,
                            sentence_response,
                            full_semantic_answer,
                            source="sentence",
                            answer_domain=resolved_answer_domain,
                            sentence_index=sent_idx,
                        )
                        semantic_sentence_infos.append(
                            _apply_clue_alpha_gate(
                                semantic_info,
                                first_token_prob=prefix_eval["first_token_prob"],
                                threshold=semantic_clue_threshold,
                                clue_alpha=clue_alpha,
                            )
                        )

                    semantic_summary = _semantic_candidate_summary(
                        [full_semantic_info] + semantic_sentence_infos
                    )
                    semantic_final_ids = semantic_summary["final_equivalent_first_token_ids"]
                    semantic_wrong_ids = semantic_summary["wrong_guess_first_token_ids"]
                    semantic_all_ids = semantic_summary["all_guess_first_token_ids"]

                    prefix_eval_entries: list[tuple[str, Optional[int], dict[str, Any]]] = [
                        ("no_cot", None, prefix_evals[0]),
                        *[
                            ("sentence", idx, prefix_eval)
                            for idx, prefix_eval in enumerate(prefix_evals[1:])
                        ],
                    ]
                    for source, sent_idx, prefix_eval in prefix_eval_entries:
                        final_mass, missing_final = _topk_prob_mass(
                            prefix_eval["topk_token_probs"],
                            semantic_final_ids,
                        )
                        wrong_mass, missing_wrong = _topk_prob_mass(
                            prefix_eval["topk_token_probs"],
                            semantic_wrong_ids,
                        )
                        all_mass, missing_all = _topk_prob_mass(
                            prefix_eval["topk_token_probs"],
                            semantic_all_ids,
                        )
                        semantic_prefix_final_masses.append(final_mass)
                        semantic_prefix_wrong_masses.append(wrong_mass)
                        semantic_prefix_all_masses.append(all_mass)

                        missing_base: dict[str, Any] = {"source": source}
                        if sent_idx is not None:
                            missing_base["sentence_index"] = sent_idx
                        if missing_final:
                            semantic_missing_final_by_prefix.append(
                                {**missing_base, "missing_first_token_ids": missing_final}
                            )
                        if missing_wrong:
                            semantic_missing_wrong_by_prefix.append(
                                {**missing_base, "missing_first_token_ids": missing_wrong}
                            )
                        if missing_all:
                            semantic_missing_all_by_prefix.append(
                                {**missing_base, "missing_first_token_ids": missing_all}
                            )

                    semantic_trace_meta = {
                        "format_version": 2,
                        "enabled": True,
                        "label_policy": "semantic_equivalence_gated_by_first_token_confidence",
                        "label_mapping": {str(k): v for k, v in SEMANTIC_GUESS_LABELS.items()},
                        "semantic_max_new_tokens": semantic_max_new_tokens,
                        "answer_domain": resolved_answer_domain,
                        "clue_alpha": float(clue_alpha),
                        "clue_alpha_threshold": semantic_clue_threshold,
                        "clue_alpha_overrides": [
                            bool(info.get("clue_alpha_overridden"))
                            for info in semantic_sentence_infos
                        ],
                        "below_clue_alpha_threshold": [
                            bool(info.get("below_clue_alpha_threshold"))
                            for info in semantic_sentence_infos
                        ],
                        "raw_semantic_labels": [
                            int(info.get("raw_semantic_label", info["semantic_label"]))
                            for info in semantic_sentence_infos
                        ],
                        "first_token_prob_source": "vllm_generation_logprobs",
                        "first_token_topk": semantic_first_token_topk,
                        "first_token_probs_are_topk_approx": True,
                        "full_cot_generated_continuation": full_semantic_response,
                        "full_cot_extracted_answer": full_semantic_answer,
                        "full_cot_first_token_id": full_semantic_info["first_token_id"],
                        "full_cot_first_token": full_semantic_info["first_token"],
                        **semantic_summary,
                    }
                    if semantic_summary["first_token_collisions"]:
                        tqdm.write(
                            f"  Q{q_id} trace {t_idx}: semantic first-token collisions="
                            f"{len(semantic_summary['first_token_collisions'])}"
                        )

                raw_deltas = []
                sentence_entries = []
                for sent_idx, span in enumerate(sentence_spans):
                    prob = prefix_probs[sent_idx + 1]
                    delta = prob - prefix_probs[sent_idx]
                    raw_deltas.append(delta)
                    pred_token_id, pred_token_prob = prefix_pred_tokens[sent_idx + 1]
                    semantic_guess = None
                    if semantic_trace_meta is not None and prefix_evals is not None:
                        semantic_final_mass = semantic_prefix_final_masses[sent_idx + 1]
                        semantic_wrong_mass = semantic_prefix_wrong_masses[sent_idx + 1]
                        semantic_all_mass = semantic_prefix_all_masses[sent_idx + 1]
                        semantic_guess = dict(semantic_sentence_infos[sent_idx])
                        first_token_id = semantic_guess.get("first_token_id")
                        prefix_eval = prefix_evals[sent_idx + 1]
                        semantic_guess.update(
                            {
                                "first_token_prob": (
                                    prefix_eval["first_token_prob"]
                                    if first_token_id == prefix_eval["first_token_id"]
                                    else 0.0
                                ),
                                "final_equivalent_first_token_prob_mass": semantic_final_mass,
                                "wrong_guess_first_token_prob_mass": semantic_wrong_mass,
                                "all_guess_first_token_prob_mass": semantic_all_mass,
                            }
                        )
                    entry = {
                        "sentence_index": sent_idx,
                        "start_pos": span.start_pos,
                        "end_pos": span.end_pos,
                        "end_token_pos": span.end_token_pos,
                        "text": span.text,
                        "actual_predicted_token": {
                            "token_id": pred_token_id,
                            "token": tokenizer.decode([pred_token_id]) if pred_token_id is not None else None,
                            "token_prob": pred_token_prob,
                        },
                        "target_token": {
                            "target_token_prob": prob,
                            "importance_delta": delta,
                        },
                    }
                    if semantic_guess is not None:
                        entry["semantic_label"] = semantic_guess["semantic_label"]
                        entry["semantic_label_name"] = semantic_guess["semantic_label_name"]
                        entry["semantic_guess"] = semantic_guess
                    sentence_entries.append(entry)

                norm = torch.softmax(torch.tensor(raw_deltas, dtype=torch.float32), dim=0).tolist()
                for entry, prob_mass in zip(sentence_entries, norm):
                    entry["target_token"]["importance_prob"] = float(prob_mass)

                if semantic_trace_meta is not None:
                    semantic_trace_meta["prefix_final_equivalent_first_token_probs"] = (
                        semantic_prefix_final_masses
                    )
                    semantic_trace_meta["prefix_wrong_guess_first_token_probs"] = (
                        semantic_prefix_wrong_masses
                    )
                    semantic_trace_meta["prefix_all_guess_first_token_probs"] = semantic_prefix_all_masses
                    semantic_trace_meta["missing_final_equivalent_first_tokens_by_prefix"] = (
                        semantic_missing_final_by_prefix
                    )
                    semantic_trace_meta["missing_wrong_guess_first_tokens_by_prefix"] = (
                        semantic_missing_wrong_by_prefix
                    )
                    semantic_trace_meta["missing_all_guess_first_tokens_by_prefix"] = (
                        semantic_missing_all_by_prefix
                    )

                elapsed = time.time() - t0
                tqdm.write(
                    f"  Q{q_id} trace {t_idx}: {len(sentence_entries)} sentences, "
                    f"target={tokenizer.decode([full_token_id])!r}, computed in {elapsed:.1f}s"
                )

                trace_results.append(
                    {
                        "trace_index": t_idx,
                        "skipped": False,
                        "n_tokens": len(full_trace_ids),
                        "target_pos": len(full_trace_ids) - 1,
                        "target_token_id": full_token_id,
                        "target_token": tokenizer.decode([full_token_id]),
                        "target_token_prob_full_cot": full_token_prob,
                        "target_token_prob_no_cot": no_cot_prob,
                        "prefix_target_probs": prefix_probs,
                        "sentence_importance_scores": raw_deltas,
                        "sentence_importance_probs": norm,
                        "sentence_end_positions": [entry["end_token_pos"] for entry in sentence_entries],
                        "sentence_spans": sentence_entries,
                        "extracted_answer": extracted_answers[t_idx] if t_idx < len(extracted_answers) else None,
                        "full_response": full_response,
                        "no_cot_response": no_cot_response,
                        "full_ids": full_trace_ids,
                        **(
                            {"semantic_guess_labels": semantic_trace_meta}
                            if semantic_trace_meta is not None
                            else {}
                        ),
                    }
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    tqdm.write(f"  [x] Q{q_id} trace {t_idx}: OOM, skipping")
                else:
                    raise
            finally:
                if reset_prefix_cache_per_trace and enable_prefix_caching:
                    _reset_prefix_cache(llm)
                gc.collect()
                torch.cuda.empty_cache()

        if not trace_results:
            tqdm.write(f"  Q{q_id}: all traces failed, skipping save")
            continue

        result = {
            "question_id": q_id,
            "input_text": input_text,
            "GT_answer": gt_answer,
            "attribution_method": "sentence_causal",
            "granularity": "sentence",
            "model": model_block,
            "segmentation": {
                "source": "decoded_cot_text",
                "saved_as": "token_positions",
            },
            "traces": trace_results,
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        tqdm.write(f"  Q{q_id}: saved {len(trace_results)} traces to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
