#!/usr/bin/env python3
"""Perturb numeric literals in CoT traces around commitment boundaries.

This experiment consumes sentence-causal attribution JSONs and asks whether
numeric corruption before/after the inferred commitment point changes the
model's own final boxed answer.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import fire
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attributions.utils import extract_boxed_answer, get_answer_suffix


PRE_BOUNDARY_EARLY_EXIT = "pre_boundary_early_exit"
CONDITIONS = ("post_boundary", "whole_cot", PRE_BOUNDARY_EARLY_EXIT)
DEFAULT_LEVELS = (0, 10, 20, 50, 80, 100)
DEFAULT_DELTAS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
DEFAULT_ABSOLUTE_BINS = (0, 10, 25, 50, 100, 200)

# Match conservative standalone integers/decimals. This still catches LaTeX
# fraction arguments such as \frac{3}{5}, while avoiding x_1 and command2.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_\\])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?![A-Za-z0-9_])"
)


@dataclass(frozen=True)
class NumericOccurrence:
    sentence_index: int
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class Replacement:
    sentence_index: int
    char_start: int
    char_end: int
    old_text: str
    new_text: str
    delta: int


@dataclass
class RunSpec:
    prompt_ids: list[int]
    row: dict[str, Any]


def _parse_int_list(value: Any, default: Optional[Sequence[int]] = None) -> Optional[list[int]]:
    if value is None:
        return list(default) if default is not None else None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return list(default) if default is not None else None
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _default_attr_dir(model: str, data_name: str) -> Path:
    return (
        Path("outputs")
        / model.split("/")[-1]
        / data_name.split("/")[-1]
        / "contribution_graphs"
        / "sentence_causal"
        / "boxed"
    )


def _default_output_file(model: str, data_name: str) -> Path:
    return (
        Path("outputs")
        / model.split("/")[-1]
        / data_name.split("/")[-1]
        / "perturbations"
        / "cot_number_perturbation.jsonl"
    )


def _question_id_from_path(path: Path) -> Optional[int]:
    match = re.search(r"question_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def _load_attr_files(attr_dir: Path, question_ids: Optional[set[int]]) -> list[Path]:
    files = sorted(attr_dir.glob("question_*.json"))
    if question_ids is not None:
        files = [
            path for path in files
            if (qid := _question_id_from_path(path)) is not None and qid in question_ids
        ]
    if not files:
        raise FileNotFoundError(f"No question_*.json attribution files found in {attr_dir}")
    return files


def _decode_token_ids(tokenizer, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _encode_text(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def _compute_commitment_boundary(prefix_final_probs: Sequence[float]) -> tuple[int, float]:
    """Return sentence index B and spike size, ignoring the no-CoT prefix."""
    sentence_probs = [float(v) for v in prefix_final_probs[1:]]
    if len(sentence_probs) < 2:
        raise ValueError("Need at least two sentence final-probability masses")
    diffs = [sentence_probs[idx + 1] - sentence_probs[idx] for idx in range(len(sentence_probs) - 1)]
    best = max(range(len(diffs)), key=lambda idx: diffs[idx])
    return best + 1, float(diffs[best])


def _numeric_occurrences(text: str, sentence_index: int, char_offset: int = 0) -> list[NumericOccurrence]:
    return [
        NumericOccurrence(
            sentence_index=sentence_index,
            char_start=char_offset + match.start(),
            char_end=char_offset + match.end(),
            text=match.group(0),
        )
        for match in NUMBER_RE.finditer(text)
    ]


def _format_perturbed_number(old_text: str, delta: int) -> str:
    if "." in old_text:
        decimals = len(old_text.rsplit(".", 1)[1])
        new_value = float(old_text) + int(delta)
        rendered = f"{new_value:.{decimals}f}"
        if old_text.startswith("+") and not rendered.startswith("-"):
            rendered = "+" + rendered
        return rendered

    new_value = int(old_text) + int(delta)
    if old_text.startswith("+") and new_value >= 0:
        return f"+{new_value}"
    return str(new_value)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def _select_occurrences(
    occurrences: Sequence[NumericOccurrence],
    level: int,
    seed: int,
) -> list[NumericOccurrence]:
    if level <= 0 or not occurrences:
        return []
    n_selected = int(round((float(level) / 100.0) * len(occurrences)))
    n_selected = max(0, min(len(occurrences), n_selected))
    rng = random.Random(seed)
    indices = list(range(len(occurrences)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:n_selected])
    return [occurrences[idx] for idx in selected_indices]


def _make_replacements(
    selected: Sequence[NumericOccurrence],
    seed: int,
    deltas: Sequence[int],
) -> list[Replacement]:
    rng = random.Random(seed)
    replacements = []
    for occurrence in selected:
        delta = int(rng.choice(list(deltas)))
        replacements.append(
            Replacement(
                sentence_index=occurrence.sentence_index,
                char_start=occurrence.char_start,
                char_end=occurrence.char_end,
                old_text=occurrence.text,
                new_text=_format_perturbed_number(occurrence.text, delta),
                delta=delta,
            )
        )
    return replacements


def _apply_replacements(cot_text: str, replacements: Sequence[Replacement]) -> str:
    mutated = cot_text
    for replacement in sorted(replacements, key=lambda item: item.char_start, reverse=True):
        mutated = (
            mutated[:replacement.char_start]
            + replacement.new_text
            + mutated[replacement.char_end:]
        )
    return mutated


def _extract_answer_from_continuation(answer_suffix: str, generated: str) -> str:
    if not generated:
        return ""
    if r"\boxed{" in answer_suffix:
        return extract_boxed_answer(answer_suffix + generated).strip()
    return generated.strip()


def _fallback_math_verify(reference: str, candidate: str) -> dict[str, Any]:
    def normalize(text: str) -> str:
        return (
            re.sub(r"\s+", "", str(text).strip().strip("$"))
            .replace(r"\tfrac", r"\frac")
            .replace(r"\dfrac", r"\frac")
            .replace(r"\left", "")
            .replace(r"\right", "")
            .casefold()
        )

    ref_norm = normalize(reference)
    cand_norm = normalize(candidate)
    if not cand_norm:
        return {"equivalent": False, "method": "no_guess", "error": None}
    if ref_norm == cand_norm:
        return {"equivalent": True, "method": "normalized_exact", "error": None}
    try:
        from math_verify import parse, verify

        def variants(text: str) -> list[str]:
            raw = str(text).strip()
            normalized = (
                raw
                .replace(r"\tfrac", r"\frac")
                .replace(r"\dfrac", r"\frac")
                .replace(r"\left", "")
                .replace(r"\right", "")
            )
            candidates = [raw, f"${raw}$", normalized, f"${normalized}$"]
            seen = set()
            return [item for item in candidates if item and not (item in seen or seen.add(item))]

        for ref_variant in variants(reference):
            ref_parsed = parse(ref_variant)
            if not ref_parsed:
                continue
            for cand_variant in variants(candidate):
                cand_parsed = parse(cand_variant)
                if cand_parsed and verify(ref_parsed, cand_parsed):
                    return {"equivalent": True, "method": "math_verify", "error": None}
        return {
            "equivalent": False,
            "method": "math_verify",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"equivalent": False, "method": "verification_error", "error": str(exc)}


def _verify_math_answer_equivalence(reference: str, candidate: str) -> dict[str, Any]:
    try:
        from attributions.vllm_sentence_causal import _verify_answer_equivalence

        return _verify_answer_equivalence(reference, candidate, "math")
    except Exception:
        return _fallback_math_verify(reference, candidate)


def _sentence_char_ranges(
    tokenizer,
    full_ids: Sequence[int],
    sentence_spans: Sequence[dict[str, Any]],
    cot_start: int,
    cot_end: int,
) -> tuple[str, list[tuple[int, int]]]:
    cot_ids = list(full_ids[cot_start:cot_end])
    cot_text = _decode_token_ids(tokenizer, cot_ids)
    ranges: list[tuple[int, int]] = []
    cursor = 0

    for span in sentence_spans:
        sentence_text = _decode_token_ids(
            tokenizer,
            full_ids[int(span["start_pos"]):int(span["end_pos"])],
        )
        found = cot_text.find(sentence_text, cursor)
        if found == -1:
            raise ValueError(
                f"Could not align sentence {span.get('sentence_index')} text inside decoded CoT"
            )
        ranges.append((found, found + len(sentence_text)))
        cursor = found + len(sentence_text)

    return cot_text, ranges


def _eligible_occurrences(
    cot_text: str,
    sentence_ranges: Sequence[tuple[int, int]],
    sentence_indices: Sequence[int],
) -> list[NumericOccurrence]:
    occurrences: list[NumericOccurrence] = []
    for sentence_index in sentence_indices:
        start, end = sentence_ranges[sentence_index]
        occurrences.extend(_numeric_occurrences(cot_text[start:end], sentence_index, start))
    return occurrences


def _condition_sentence_indices(
    condition: str,
    commitment_boundary_idx: int,
    n_sentences: int,
) -> list[int]:
    if condition == "whole_cot":
        return list(range(n_sentences))
    if condition == "post_boundary":
        return list(range(commitment_boundary_idx + 1, n_sentences))
    if condition == PRE_BOUNDARY_EARLY_EXIT:
        return list(range(commitment_boundary_idx + 1))
    raise ValueError(f"Unknown perturbation condition: {condition}")


def _trace_skip_reason(trace: dict[str, Any]) -> Optional[str]:
    if trace.get("skipped"):
        return str(trace.get("skip_reason") or "trace_marked_skipped")
    semantic_meta = trace.get("semantic_guess_labels") or {}
    if semantic_meta.get("first_token_collisions"):
        return "semantic_first_token_collisions"
    if not semantic_meta.get("prefix_final_equivalent_first_token_probs"):
        return "missing_prefix_final_equivalent_first_token_probs"
    if not semantic_meta.get("full_cot_extracted_answer"):
        return "missing_full_cot_extracted_answer"
    if not trace.get("sentence_spans"):
        return "missing_sentence_spans"
    if not trace.get("full_ids"):
        return "missing_full_ids"
    return None


def _base_row(
    *,
    model: str,
    data_name: str,
    attr_file: Path,
    question_id: int,
    trace: dict[str, Any],
    commitment_boundary_idx: int,
    commitment_spike: float,
) -> dict[str, Any]:
    sentence_spans = trace["sentence_spans"]
    semantic_meta = trace["semantic_guess_labels"]
    return {
        "status": "ok",
        "model": model,
        "data_name": data_name,
        "attr_file": str(attr_file),
        "question_id": question_id,
        "trace_index": int(trace.get("trace_index", -1)),
        "commitment_boundary_idx": int(commitment_boundary_idx),
        "boundary_sentence_index": int(commitment_boundary_idx),
        "commitment_boundary_pos": int(sentence_spans[commitment_boundary_idx]["end_pos"]),
        "boundary_end_pos": int(sentence_spans[commitment_boundary_idx]["end_pos"]),
        "commitment_spike": float(commitment_spike),
        "original_full_cot_answer": semantic_meta.get("full_cot_extracted_answer", ""),
        "target_token_id": trace.get("target_token_id"),
        "target_token": trace.get("target_token"),
    }


def _build_run_specs_for_trace(
    *,
    tokenizer,
    model: str,
    data_name: str,
    attr_file: Path,
    question_id: int,
    trace: dict[str, Any],
    levels: Sequence[int],
    repeats: int,
    deltas: Sequence[int],
    base_seed: int,
) -> tuple[list[RunSpec], Optional[dict[str, Any]]]:
    skip_reason = _trace_skip_reason(trace)
    trace_index = int(trace.get("trace_index", -1))
    if skip_reason is not None:
        return [], {
            "status": "skipped",
            "skip_reason": skip_reason,
            "model": model,
            "data_name": data_name,
            "attr_file": str(attr_file),
            "question_id": question_id,
            "trace_index": trace_index,
        }

    sentence_spans = trace["sentence_spans"]
    full_ids = [int(token_id) for token_id in trace["full_ids"]]
    semantic_meta = trace["semantic_guess_labels"]
    try:
        commitment_boundary_idx, commitment_spike = _compute_commitment_boundary(
            semantic_meta["prefix_final_equivalent_first_token_probs"]
        )
    except ValueError as exc:
        return [], {
            "status": "skipped",
            "skip_reason": str(exc),
            "model": model,
            "data_name": data_name,
            "attr_file": str(attr_file),
            "question_id": question_id,
            "trace_index": trace_index,
        }

    if commitment_boundary_idx >= len(sentence_spans) - 1:
        return [], {
            "status": "skipped",
            "skip_reason": "commitment_boundary_is_final_sentence",
            "model": model,
            "data_name": data_name,
            "attr_file": str(attr_file),
            "question_id": question_id,
            "trace_index": trace_index,
            "commitment_boundary_idx": commitment_boundary_idx,
        }

    cot_start = int(sentence_spans[0]["start_pos"])
    cot_end = int(sentence_spans[-1]["end_pos"])
    prefix_ids = full_ids[:cot_start]
    tail_ids = full_ids[cot_end:]
    try:
        cot_text, sentence_ranges = _sentence_char_ranges(
            tokenizer,
            full_ids,
            sentence_spans,
            cot_start,
            cot_end,
        )
    except ValueError as exc:
        return [], {
            "status": "skipped",
            "skip_reason": f"sentence_alignment_failed: {exc}",
            "model": model,
            "data_name": data_name,
            "attr_file": str(attr_file),
            "question_id": question_id,
            "trace_index": trace_index,
            "commitment_boundary_idx": commitment_boundary_idx,
            "boundary_sentence_index": commitment_boundary_idx,
        }

    sentence_indices_by_condition = {
        condition: _condition_sentence_indices(
            condition,
            commitment_boundary_idx,
            len(sentence_spans),
        )
        for condition in CONDITIONS
    }
    occurrences_by_condition = {
        condition: _eligible_occurrences(cot_text, sentence_ranges, sentence_indices)
        for condition, sentence_indices in sentence_indices_by_condition.items()
    }
    for condition, occurrences in occurrences_by_condition.items():
        if not occurrences:
            return [], {
                "status": "skipped",
                "skip_reason": f"no_eligible_numeric_literals_for_{condition}",
                "model": model,
                "data_name": data_name,
                "attr_file": str(attr_file),
                "question_id": question_id,
                "trace_index": trace_index,
                "commitment_boundary_idx": commitment_boundary_idx,
            }

    specs: list[RunSpec] = []
    boundary_char_end = sentence_ranges[commitment_boundary_idx][1]
    prompt_text_by_condition = {
        "post_boundary": cot_text,
        "whole_cot": cot_text,
        PRE_BOUNDARY_EARLY_EXIT: cot_text[:boundary_char_end],
    }
    early_exit_by_condition = {
        "post_boundary": False,
        "whole_cot": False,
        PRE_BOUNDARY_EARLY_EXIT: True,
    }
    for condition in CONDITIONS:
        occurrences = occurrences_by_condition[condition]
        prompt_cot_text = prompt_text_by_condition[condition]
        is_early_exit = early_exit_by_condition[condition]
        for level in levels:
            active_repeats = 1 if int(level) == 0 else int(repeats)
            for repeat in range(active_repeats):
                run_seed = _stable_seed(base_seed, question_id, trace_index, condition, int(level), repeat)
                selected = _select_occurrences(occurrences, int(level), run_seed)
                replacement_seed = _stable_seed(run_seed, "deltas")
                replacements = _make_replacements(selected, replacement_seed, deltas)
                if replacements:
                    mutated_cot_text = _apply_replacements(prompt_cot_text, replacements)
                    prompt_ids = prefix_ids + _encode_text(tokenizer, mutated_cot_text) + tail_ids
                elif is_early_exit:
                    prompt_ids = prefix_ids + _encode_text(tokenizer, prompt_cot_text) + tail_ids
                else:
                    prompt_ids = full_ids

                row = _base_row(
                    model=model,
                    data_name=data_name,
                    attr_file=attr_file,
                    question_id=question_id,
                    trace=trace,
                    commitment_boundary_idx=commitment_boundary_idx,
                    commitment_spike=commitment_spike,
                )
                row.update(
                    {
                        "condition": condition,
                        "level": int(level),
                        "repeat": int(repeat),
                        "seed": int(run_seed),
                        "eligible_number_count": len(occurrences),
                        "perturbed_number_count": len(replacements),
                        "replacements": [asdict(replacement) for replacement in replacements],
                        "is_zero_percent_baseline": int(level) == 0,
                        "early_exit": is_early_exit,
                        "kept_sentence_start": 0 if is_early_exit else None,
                        "kept_sentence_end": commitment_boundary_idx if is_early_exit else None,
                        "dropped_post_boundary": is_early_exit,
                        "cot_start": cot_start,
                        "cot_end": cot_end,
                        "prompt_cot_char_start": 0,
                        "prompt_cot_char_end": boundary_char_end if is_early_exit else len(cot_text),
                    }
                )
                specs.append(RunSpec(prompt_ids=prompt_ids, row=row))

    return specs, None


def _read_question_file(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _prepare_run_specs(
    *,
    tokenizer,
    model: str,
    data_name: str,
    attr_dir: Path,
    question_ids: Optional[set[int]],
    trace_indices: Optional[set[int]],
    levels: Sequence[int],
    repeats: int,
    deltas: Sequence[int],
    seed: int,
) -> tuple[list[RunSpec], list[dict[str, Any]], dict[str, int]]:
    specs: list[RunSpec] = []
    status_rows: list[dict[str, Any]] = []
    stats = {"files": 0, "traces_seen": 0, "traces_ready": 0, "traces_skipped": 0, "runs": 0}

    for attr_file in _load_attr_files(attr_dir, question_ids):
        stats["files"] += 1
        attr_data = _read_question_file(attr_file)
        question_id = int(attr_data.get("question_id", _question_id_from_path(attr_file) or -1))
        for trace in attr_data.get("traces", []):
            trace_index = int(trace.get("trace_index", -1))
            if trace_indices is not None and trace_index not in trace_indices:
                continue
            stats["traces_seen"] += 1
            trace_specs, skip_row = _build_run_specs_for_trace(
                tokenizer=tokenizer,
                model=model,
                data_name=data_name,
                attr_file=attr_file,
                question_id=question_id,
                trace=trace,
                levels=levels,
                repeats=repeats,
                deltas=deltas,
                base_seed=seed,
            )
            if skip_row is not None:
                status_rows.append(skip_row)
                stats["traces_skipped"] += 1
            else:
                specs.extend(trace_specs)
                stats["traces_ready"] += 1

    stats["runs"] = len(specs)
    return specs, status_rows, stats


def _continuation_text(output, tokenizer) -> str:
    completion = output.outputs[0]
    text = getattr(completion, "text", None)
    if text is not None:
        return text
    token_ids = getattr(completion, "token_ids", None) or []
    return _decode_token_ids(tokenizer, token_ids)


def _generate_rows(
    *,
    llm,
    tokenizer,
    tokens_prompt_cls,
    sampling_params,
    specs: Sequence[RunSpec],
    answer_suffix: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(specs), batch_size), desc="Perturbation batches", unit="batch"):
        batch = specs[start:start + batch_size]
        prompts = [tokens_prompt_cls(prompt_token_ids=spec.prompt_ids) for spec in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for spec, output in zip(batch, outputs):
            continuation = _continuation_text(output, tokenizer)
            extracted = _extract_answer_from_continuation(answer_suffix, continuation)
            equivalence = _verify_math_answer_equivalence(
                spec.row["original_full_cot_answer"],
                extracted,
            )
            row = dict(spec.row)
            row.update(
                {
                    "generated_continuation": continuation,
                    "perturbed_extracted_answer": extracted,
                    "equivalent_to_original_full_cot_answer": bool(equivalence["equivalent"]),
                    "equivalence_method": equivalence.get("method"),
                    "equivalence_error": equivalence.get("error"),
                }
            )
            rows.append(row)
    return rows


def _attach_baseline_reproduction(rows: list[dict[str, Any]]) -> None:
    baseline_by_trace: dict[tuple[int, int], bool] = {}
    boundary_prefix_by_trace: dict[tuple[int, int], bool] = {}
    for row in rows:
        if row.get("level") != 0:
            continue
        key = (int(row["question_id"]), int(row["trace_index"]))
        equivalent = bool(row.get("equivalent_to_original_full_cot_answer"))
        if row.get("condition") != PRE_BOUNDARY_EARLY_EXIT:
            baseline_by_trace.setdefault(key, equivalent)
        else:
            boundary_prefix_by_trace.setdefault(key, equivalent)

    for row in rows:
        key = (int(row["question_id"]), int(row["trace_index"]))
        row["baseline_reproduced"] = bool(baseline_by_trace.get(key, False))
        row["boundary_prefix_reproduced"] = bool(boundary_prefix_by_trace.get(key, False))


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) + z2 / (4 * total)) / total) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _valid_recap_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if not row.get("baseline_reproduced"):
            continue
        if row.get("condition") == PRE_BOUNDARY_EARLY_EXIT and not row.get("boundary_prefix_reproduced"):
            continue
        valid.append(row)
    return valid


def _unique_trace_count(rows: Sequence[dict[str, Any]], predicate) -> int:
    keys = {
        (int(row["question_id"]), int(row["trace_index"]))
        for row in rows
        if predicate(row)
    }
    return len(keys)


def _summarize_group(group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = [float(bool(row.get("equivalent_to_original_full_cot_answer"))) for row in group_rows]
    n_success = int(sum(successes))
    ci_low, ci_high = _wilson_ci(n_success, len(group_rows))
    perturbed = [float(row.get("perturbed_number_count", 0)) for row in group_rows]
    eligible = [float(row.get("eligible_number_count", 0)) for row in group_rows]
    return {
        "n_runs": len(group_rows),
        "n_traces": _unique_trace_count(group_rows, lambda _row: True),
        "n_success": n_success,
        "success_rate": _mean(successes),
        "success_ci95_low": ci_low,
        "success_ci95_high": ci_high,
        "avg_perturbed_numbers": _mean(perturbed),
        "avg_eligible_numbers": _mean(eligible),
    }


def _summary_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in _valid_recap_rows(rows):
        key = (str(row["condition"]), int(row["level"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    condition_order = {condition: idx for idx, condition in enumerate(CONDITIONS)}
    for (condition, level), group_rows in sorted(
        grouped.items(),
        key=lambda item: (condition_order.get(item[0][0], 999), item[0][1]),
    ):
        summary.append(
            {
                "condition": condition,
                "level": level,
                **_summarize_group(group_rows),
            }
        )
    return summary


def _absolute_bin_label(count: int, upper_bounds: Sequence[int]) -> str:
    bounds = sorted({int(bound) for bound in upper_bounds if int(bound) >= 0})
    if not bounds or bounds[0] != 0:
        bounds = [0, *bounds]
    if count <= 0:
        return "0"
    lower = 1
    for upper in bounds:
        if upper == 0:
            continue
        if count <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def _absolute_bin_sort_key(label: str) -> int:
    if label == "0":
        return 0
    return int(label.rstrip("+").split("-", 1)[0])


def _absolute_summary_groups(
    rows: Sequence[dict[str, Any]],
    absolute_bins: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _valid_recap_rows(rows):
        label = _absolute_bin_label(int(row.get("perturbed_number_count", 0)), absolute_bins)
        key = (str(row["condition"]), label)
        grouped.setdefault(key, []).append(row)

    summary = []
    condition_order = {condition: idx for idx, condition in enumerate(CONDITIONS)}
    for (condition, bin_label), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            condition_order.get(item[0][0], 999),
            _absolute_bin_sort_key(item[0][1]),
        ),
    ):
        summary.append(
            {
                "condition": condition,
                "perturbed_count_bin": bin_label,
                **_summarize_group(group_rows),
            }
        )
    return summary


def _format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.{digits}f}"


def _print_table(title: str, summary: Sequence[dict[str, Any]], group_headers: Sequence[str]) -> None:
    if not summary:
        print(f"\n{title}")
        print("No rows available.")
        return

    headers = [
        *group_headers,
        "n_runs",
        "n_traces",
        "n_success",
        "success_rate",
        "success_ci95_low",
        "success_ci95_high",
        "avg_perturbed_numbers",
        "avg_eligible_numbers",
    ]
    formatted_rows = []
    for item in summary:
        row = {header: str(item[header]) for header in group_headers}
        if "level" in row:
            row["level"] = f"{float(item['level']):.1f}"
        row.update(
            {
                "n_runs": str(item["n_runs"]),
                "n_traces": str(item["n_traces"]),
                "n_success": str(item["n_success"]),
                "success_rate": _format_float(item["success_rate"]),
                "success_ci95_low": _format_float(item["success_ci95_low"]),
                "success_ci95_high": _format_float(item["success_ci95_high"]),
                "avg_perturbed_numbers": _format_float(item["avg_perturbed_numbers"]),
                "avg_eligible_numbers": _format_float(item["avg_eligible_numbers"]),
            }
        )
        formatted_rows.append(row)

    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }
    print(f"\n{title}")
    print(" ".join(header.rjust(widths[header]) for header in headers))
    for row in formatted_rows:
        print(" ".join(row[header].rjust(widths[header]) for header in headers))


def _print_recap(rows: Sequence[dict[str, Any]], absolute_bins: Sequence[int]) -> None:
    percentage_summary = _summary_groups(rows)
    absolute_summary = _absolute_summary_groups(rows, absolute_bins)
    if not percentage_summary:
        print("\nNo recap rows available after baseline filters.")
        return

    _print_table("Percentage recap table", percentage_summary, ["condition", "level"])
    _print_table(
        "Absolute perturbation-count recap table",
        absolute_summary,
        ["condition", "perturbed_count_bin"],
    )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    skipped_rows = [row for row in rows if row.get("status") == "skipped"]
    baseline_failed = _unique_trace_count(
        ok_rows,
        lambda row: not bool(row.get("baseline_reproduced")),
    )
    boundary_prefix_failed = _unique_trace_count(
        ok_rows,
        lambda row: (
            row.get("condition") == PRE_BOUNDARY_EARLY_EXIT
            and not bool(row.get("boundary_prefix_reproduced"))
        ),
    )
    print()
    print(f"Skipped rows: {len(skipped_rows)}")
    print(f"Baseline failed traces: {baseline_failed}")
    print(f"Boundary-prefix failed traces: {boundary_prefix_failed}")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    model: str = "openai/gpt-oss-20b",
    data_name: str = "opencompass/AIME2025",
    question_ids: Optional[Sequence[int]] = None,
    trace_indices: Optional[Sequence[int]] = None,
    attr_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    perturbation_levels: Sequence[int] = DEFAULT_LEVELS,
    absolute_bins: Sequence[int] = DEFAULT_ABSOLUTE_BINS,
    repeats: int = 3,
    deltas: Sequence[int] = DEFAULT_DELTAS,
    max_new_tokens: int = 64,
    seed: int = 0,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    enable_prefix_caching: bool = True,
):
    """Run or dry-run the CoT number perturbation experiment."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parsed_question_ids = set(_parse_int_list(question_ids) or []) if question_ids is not None else None
    parsed_trace_indices = set(_parse_int_list(trace_indices) or []) if trace_indices is not None else None
    levels = _parse_int_list(perturbation_levels, DEFAULT_LEVELS) or list(DEFAULT_LEVELS)
    parsed_absolute_bins = _parse_int_list(absolute_bins, DEFAULT_ABSOLUTE_BINS) or list(DEFAULT_ABSOLUTE_BINS)
    parsed_deltas = _parse_int_list(deltas, DEFAULT_DELTAS) or list(DEFAULT_DELTAS)
    resolved_attr_dir = Path(attr_dir) if attr_dir else _default_attr_dir(model, data_name)
    resolved_output_file = Path(output_file) if output_file else _default_output_file(model, data_name)

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    specs, status_rows, stats = _prepare_run_specs(
        tokenizer=tokenizer,
        model=model,
        data_name=data_name,
        attr_dir=resolved_attr_dir,
        question_ids=parsed_question_ids,
        trace_indices=parsed_trace_indices,
        levels=levels,
        repeats=repeats,
        deltas=parsed_deltas,
        seed=seed,
    )
    print(json.dumps({"dry_run": bool(dry_run), **stats}, indent=2))
    if dry_run:
        return {"stats": stats, "status_rows": status_rows[:20]}

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm_kwargs = dict(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        seed=seed,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if "ministral" in model.lower() or "mistral" in model.lower():
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"

    print(f"Loading model with vLLM: {model}")
    llm = LLM(**llm_kwargs)
    vllm_tokenizer = llm.get_tokenizer()
    answer_suffix = get_answer_suffix(data_name)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        seed=seed,
    )
    # Rebuild with the vLLM tokenizer to avoid tiny tokenizer-version drift.
    specs, status_rows, stats = _prepare_run_specs(
        tokenizer=vllm_tokenizer,
        model=model,
        data_name=data_name,
        attr_dir=resolved_attr_dir,
        question_ids=parsed_question_ids,
        trace_indices=parsed_trace_indices,
        levels=levels,
        repeats=repeats,
        deltas=parsed_deltas,
        seed=seed,
    )
    rows = _generate_rows(
        llm=llm,
        tokenizer=vllm_tokenizer,
        tokens_prompt_cls=TokensPrompt,
        sampling_params=sampling_params,
        specs=specs,
        answer_suffix=answer_suffix,
        batch_size=batch_size,
    )
    _attach_baseline_reproduction(rows)
    all_rows = [*status_rows, *rows]
    _write_jsonl(resolved_output_file, all_rows, overwrite=overwrite)
    _print_recap(all_rows, parsed_absolute_bins)
    print(f"Wrote {len(all_rows)} rows to {resolved_output_file}")
    return {
        "stats": stats,
        "output_file": str(resolved_output_file),
        "rows": len(all_rows),
        "summary": _summary_groups(all_rows),
        "absolute_summary": _absolute_summary_groups(all_rows, parsed_absolute_bins),
    }


if __name__ == "__main__":
    fire.Fire(main)
