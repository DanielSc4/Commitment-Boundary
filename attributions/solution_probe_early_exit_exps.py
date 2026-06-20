#!/usr/bin/env python3
"""
Focused early-exit experiments for the three-way solution probe.

This intentionally avoids the old compression experiment registry. It runs:
  - full-CoT baseline
  - no-CoT baseline
  - fixed-percentage sentence early-exit baselines
  - probe early-exit variants for k consecutive class-2 predictions

The only supported probe task is train_solution_probe.py's three_way mode:
  0 = no usable or not-yet-confident guess
  1 = confident wrong/mid guess
  2 = confident final-equivalent guess
"""

import gc
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fire
import torch
from nnsight import LanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attributions.modeling import (
    collect_residual_stream,
    detect_model_config,
    greedy_generate,
    load_nnsight_model,
)
from attributions.trace_context import build_trace_context
from attributions.train_solution_probe import (
    CONTEXT_POLICY,
    SentenceCausalProbe,
    _normalize_optional_positive_int,
    load_solution_attribution_data,
)
from attributions.utils import (
    assemble_solution_for_eval,
    extract_boxed_answer,
    get_answer_suffix,
    get_code_benchmark_metadata,
    get_reasoning_traces,
    get_thinking_tokens,
    is_code_benchmark,
    load_question_split,
    parse_code_eval_entry,
    run_code_eval,
)


def check_correct(generated: str, gt_answer: str) -> bool:
    """Compare a generated boxed answer with the benchmark answer."""
    extracted = extract_boxed_answer(generated)
    candidate = extracted if extracted else generated
    return str(gt_answer).strip().casefold() in candidate.strip().casefold()


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> Optional[List[float]]:
    if total <= 0:
        return None
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) + z2 / (4 * total)) / total) ** 0.5 / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _bootstrap_ci(values: List[float], reducer, n_boot: int = 1000, seed: int = 0) -> Optional[List[float]]:
    if not values:
        return None
    vals = [float(v) for v in values]
    if len(vals) == 1:
        point = float(reducer(vals))
        return [point, point]

    rng = random.Random(seed)
    n = len(vals)
    boot = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boot.append(float(reducer(sample)))
    boot.sort()

    def percentile(sorted_vals: List[float], q: float) -> float:
        pos = (len(sorted_vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    return [percentile(boot, 0.025), percentile(boot, 0.975)]


@dataclass
class VariantAccumulator:
    name: str
    n_total: int = 0
    n_correct: int = 0
    saved_fractions: List[float] = field(default_factory=list)
    kept_tokens: List[int] = field(default_factory=list)
    total_tokens: List[int] = field(default_factory=list)
    exit_sentence_fracs: List[float] = field(default_factory=list)
    gen_tokens: List[int] = field(default_factory=list)
    seconds: float = 0.0
    n_fallback_full: int = 0

    def add(self, result: dict) -> None:
        if result.get("skipped") or result.get("correct") is None:
            return
        self.n_total += 1
        self.n_correct += int(bool(result["correct"]))
        self.saved_fractions.append(float(result.get("saved_fraction", 0.0)))
        self.kept_tokens.append(int(result.get("n_cot_tokens_kept", 0)))
        self.total_tokens.append(int(result.get("n_cot_tokens_total", 0)))
        if result.get("exit_sentence_frac") is not None:
            self.exit_sentence_fracs.append(float(result["exit_sentence_frac"]))
        self.gen_tokens.append(int(result.get("n_gen_tokens", 0)))
        self.seconds += float(result.get("seconds", 0.0))
        self.n_fallback_full += int(bool(result.get("fallback_full", False)))

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    def summary(self) -> dict:
        def mean(vals: List[float]) -> Optional[float]:
            return sum(vals) / len(vals) if vals else None

        def median(vals: List[float]) -> Optional[float]:
            return float(statistics.median(vals)) if vals else None

        return {
            "accuracy": self.accuracy,
            "accuracy_ci": _wilson_ci(self.n_correct, self.n_total),
            "n_correct": self.n_correct,
            "n_total": self.n_total,
            "mean_cot_tokens_kept": mean(self.kept_tokens),
            "mean_cot_tokens_total": mean(self.total_tokens),
            "mean_tokens_saved": (
                mean([t - k for t, k in zip(self.total_tokens, self.kept_tokens)])
                if self.total_tokens
                else None
            ),
            "mean_saved_fraction": mean(self.saved_fractions),
            "mean_saved_fraction_ci": _bootstrap_ci(self.saved_fractions, mean),
            "median_saved_fraction": median(self.saved_fractions),
            "median_saved_fraction_ci": _bootstrap_ci(self.saved_fractions, median),
            "mean_exit_sentence_frac": mean(self.exit_sentence_fracs),
            "mean_generation_tokens": mean(self.gen_tokens),
            "seconds_total": self.seconds,
            "seconds_per_trace": self.seconds / self.n_total if self.n_total else None,
            "n_fallback_full": self.n_fallback_full,
            "fallback_full_rate": self.n_fallback_full / self.n_total if self.n_total else None,
            "fallback_full_rate_ci": _wilson_ci(self.n_fallback_full, self.n_total),
        }


def _parse_int_list(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _parse_float_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def _percentage_tag(percent: float) -> str:
    return f"{percent:g}".replace(".", "p")


def _threshold_tag(value: Optional[float]) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _early_exit_name(k: int, class2_prob_threshold: Optional[float], class2_margin_threshold: Optional[float]) -> str:
    name = f"early_exit_k{k}"
    if class2_prob_threshold is not None:
        name += f"_p2{_threshold_tag(class2_prob_threshold)}"
    if class2_margin_threshold is not None:
        name += f"_m{_threshold_tag(class2_margin_threshold)}"
    return name


def _default_attr_dir(model: str, data_name: str) -> Path:
    return (
        Path("outputs")
        / model.split("/")[-1]
        / data_name.split("/")[-1]
        / "contribution_graphs"
        / "sentence_causal"
        / "boxed"
    )


def _load_probe_results(probe_dir: Path, layer: int) -> Tuple[dict, Path, Path]:
    results_path = probe_dir / f"solution_probe_results_L{layer}.json"
    weights_path = probe_dir / f"solution_probe_weights_L{layer}.pt"
    if not results_path.exists():
        raise FileNotFoundError(f"Solution probe results not found: {results_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Solution probe weights not found: {weights_path}")
    with open(results_path) as f:
        results = json.load(f)
    task_mode = results.get("hyperparams", {}).get("task_mode")
    if task_mode != "three_way":
        raise ValueError(
            f"This script only supports three_way solution probes; "
            f"{results_path} has task_mode={task_mode!r}"
        )
    return results, results_path, weights_path


def _load_split(
    probe_dir: Path,
    attr_dir: Path,
    split_file: Optional[str],
) -> Tuple[Optional[dict], Optional[Path]]:
    candidates = []
    if split_file:
        candidates.append(Path(split_file))
    candidates.extend([
        probe_dir / "question_split.json",
        probe_dir.parent / "question_split.json",
        attr_dir / "question_split.json",
    ])
    for path in candidates:
        if path.exists():
            return load_question_split(path), path
    return None, None


def _select_question_ids(
    n_questions: int,
    split: Optional[dict],
    subset: str,
    probe_train_data_name: str,
    eval_data_name: str,
) -> List[int]:
    if subset == "auto":
        subset = "eval" if split is not None and probe_train_data_name == eval_data_name else "all"
    if subset == "all" or split is None:
        return list(range(n_questions))
    if subset not in {"eval", "probe_train", "probe_val", "probe_test"}:
        raise ValueError("eval_question_subset must be auto|all|eval|probe_train|probe_val|probe_test")
    return [q for q in split[subset] if 0 <= q < n_questions]


def _tokens_to_text(tokenizer, token_ids: Iterable[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _generate(
    lm: LanguageModel,
    tokenizer,
    ids: List[int],
    max_new_tokens: int,
    device: str,
) -> Tuple[str, int]:
    ids_t = torch.tensor([ids], device=device)
    gen_ids = greedy_generate(lm, ids_t, max_new_tokens, tokenizer.eos_token_id)
    return _tokens_to_text(tokenizer, gen_ids), len(gen_ids)


def _run_full_baseline(ctx, lm, tokenizer, max_new_tokens: int, device: str, is_code: bool) -> dict:
    t0 = time.time()
    generated, n_gen = _generate(lm, tokenizer, list(ctx.full_ids), max_new_tokens, device)
    correct = None if is_code else check_correct(generated, ctx.gt_answer)
    n_total = ctx.cot_end - ctx.cot_start
    return {
        "experiment": "baseline_full",
        "correct": correct,
        "generated": generated,
        "n_gen_tokens": n_gen,
        "n_cot_tokens_kept": n_total,
        "n_cot_tokens_total": n_total,
        "saved_fraction": 0.0,
        "seconds": time.time() - t0,
    }


def _run_nocot_baseline(ctx, lm, tokenizer, max_new_tokens: int, device: str, is_code: bool) -> dict:
    t0 = time.time()
    prompt = list(ctx.full_ids[:ctx.prompt_len])
    start_think = list(ctx.full_ids[ctx.cot_start:ctx.cot_start + ctx.n_start_think])
    end_think = list(ctx.full_ids[ctx.cot_end - ctx.n_end_think:ctx.cot_end])
    suffix = list(ctx.full_ids[ctx.suffix_start:])
    ids = prompt + start_think + end_think + suffix
    generated, n_gen = _generate(lm, tokenizer, ids, max_new_tokens, device)
    correct = None if is_code else check_correct(generated, ctx.gt_answer)
    n_total = ctx.cot_end - ctx.cot_start
    return {
        "experiment": "baseline_nocot",
        "correct": correct,
        "generated": generated,
        "n_gen_tokens": n_gen,
        "n_cot_tokens_kept": len(start_think) + len(end_think),
        "n_cot_tokens_total": n_total,
        "saved_fraction": max(0.0, (n_total - len(start_think) - len(end_think)) / max(n_total, 1)),
        "seconds": time.time() - t0,
    }


def _run_fixed_percentage_exit(
    ctx,
    lm,
    tokenizer,
    trace_label: dict,
    percent: float,
    max_new_tokens: int,
    device: str,
    is_code: bool,
    full_result: dict,
    nocot_result: dict,
) -> dict:
    t0 = time.time()
    name = f"baseline_fixed_p{_percentage_tag(percent)}"
    if not (0.0 <= percent <= 100.0):
        raise ValueError(f"fixed_exit_percentages values must be in [0, 100], got {percent}")
    if percent <= 0.0:
        result = dict(nocot_result)
        result.update({
            "experiment": name,
            "fixed_exit_percent": percent,
            "exit_sentence_idx": None,
            "exit_sentence_frac": 0.0,
            "target_exit_sentence_frac": percent / 100.0,
            "fallback_full": False,
            "seconds": time.time() - t0,
        })
        return result
    if percent >= 100.0:
        result = dict(full_result)
        result.update({
            "experiment": name,
            "fixed_exit_percent": percent,
            "exit_sentence_idx": None,
            "exit_sentence_frac": 1.0,
            "target_exit_sentence_frac": percent / 100.0,
            "fallback_full": False,
            "seconds": time.time() - t0,
        })
        return result

    spans = sorted(trace_label.get("unit_spans") or [], key=lambda span: int(span["end_pos"]))
    if not spans:
        result = dict(full_result)
        result.update({
            "experiment": name,
            "fixed_exit_percent": percent,
            "exit_sentence_idx": None,
            "exit_sentence_frac": None,
            "target_exit_sentence_frac": percent / 100.0,
            "fallback_full": True,
            "seconds": time.time() - t0,
        })
        return result

    n_keep = min(len(spans), max(1, math.ceil(len(spans) * percent / 100.0)))
    sent = spans[n_keep - 1]
    exit_end_pos = min(int(sent["end_pos"]), ctx.cot_end - ctx.n_end_think)
    end_think = list(ctx.full_ids[ctx.cot_end - ctx.n_end_think:ctx.cot_end])
    suffix = list(ctx.full_ids[ctx.suffix_start:])
    ids = list(ctx.full_ids[:exit_end_pos]) + end_think + suffix
    generated, n_gen = _generate(lm, tokenizer, ids, max_new_tokens, device)
    correct = None if is_code else check_correct(generated, ctx.gt_answer)
    n_total = ctx.cot_end - ctx.cot_start
    n_kept = max(0, exit_end_pos - ctx.cot_start) + ctx.n_end_think
    saved = max(0, n_total - n_kept)
    return {
        "experiment": name,
        "fixed_exit_percent": percent,
        "correct": correct,
        "generated": generated,
        "n_gen_tokens": n_gen,
        "n_cot_tokens_kept": n_kept,
        "n_cot_tokens_total": n_total,
        "tokens_saved": saved,
        "saved_fraction": saved / max(n_total, 1),
        "fallback_full": False,
        "exit_sentence_idx": int(sent.get("sentence_index", n_keep - 1)),
        "exit_sentence_frac": n_keep / max(len(spans), 1),
        "target_exit_sentence_frac": percent / 100.0,
        "trigger_sentence": sent,
        "seconds": time.time() - t0,
    }


def _score_sentences(
    probe: SentenceCausalProbe,
    full_resid: torch.Tensor,
    layer: int,
    trace_label: dict,
    sentence_aggregation: str,
    device: str,
    max_sentences: Optional[int],
    max_probe_input_tokens: Optional[int],
) -> List[dict]:
    spans = trace_label["unit_spans"]
    labels = trace_label["three_way_labels"]
    kept_indices = list(range(len(spans)))
    if max_sentences is not None and len(kept_indices) > max_sentences:
        kept_indices = kept_indices[-max_sentences:]

    layer_acts = full_resid[layer].float()
    outputs = []

    with torch.no_grad():
        if sentence_aggregation == "full":
            kept_spans = [spans[i] for i in kept_indices]
            if not kept_spans:
                return []
            start_offset = int(kept_spans[0]["start_pos"])
            end_offset = int(kept_spans[-1]["end_pos"])
            if max_probe_input_tokens is not None and end_offset - start_offset > max_probe_input_tokens:
                start_offset = max(start_offset, end_offset - max_probe_input_tokens)
            acts = layer_acts[start_offset:end_offset].unsqueeze(0).to(device)
            logits_all = probe(
                acts,
                causal_window=max_probe_input_tokens,
            ).squeeze(0).cpu()
            for idx in kept_indices:
                span = spans[idx]
                end_idx = int(span["end_pos"]) - 1 - start_offset
                if end_idx < 0 or end_idx >= len(logits_all):
                    continue
                logits = logits_all[end_idx].float()
                probs = torch.softmax(logits, dim=-1)
                pred = int(logits.argmax().item())
                outputs.append({
                    "sentence_index": int(span.get("sentence_index", idx)),
                    "start_pos": int(span["start_pos"]),
                    "end_pos": int(span["end_pos"]),
                    "text": span.get("text", ""),
                    "label": int(labels[idx]),
                    "pred": pred,
                    "logits": [float(x) for x in logits.tolist()],
                    "probs": [float(x) for x in probs.tolist()],
                })
            return outputs

        states = []
        state_indices = []
        for idx in kept_indices:
            span = spans[idx]
            start = int(span["start_pos"])
            end = int(span["end_pos"])
            if start >= end or start < 0 or end > layer_acts.shape[0]:
                continue
            if sentence_aggregation == "last":
                states.append(layer_acts[end - 1])
            elif sentence_aggregation == "avg":
                states.append(layer_acts[start:end].mean(dim=0))
            else:
                raise ValueError(f"Unsupported sentence_aggregation={sentence_aggregation!r}")
            state_indices.append(idx)

        if not states:
            return []
        logits_seq = probe(
            torch.stack(states).unsqueeze(0).to(device),
            causal_window=max_probe_input_tokens,
        ).squeeze(0).cpu()
        for local_idx, idx in enumerate(state_indices):
            span = spans[idx]
            logits = logits_seq[local_idx].float()
            probs = torch.softmax(logits, dim=-1)
            pred = int(logits.argmax().item())
            outputs.append({
                "sentence_index": int(span.get("sentence_index", idx)),
                "start_pos": int(span["start_pos"]),
                "end_pos": int(span["end_pos"]),
                "text": span.get("text", ""),
                "label": int(labels[idx]),
                "pred": pred,
                "logits": [float(x) for x in logits.tolist()],
                "probs": [float(x) for x in probs.tolist()],
            })
    return outputs


def _passes_class2_gate(
    sent: dict,
    class2_prob_threshold: Optional[float] = None,
    class2_margin_threshold: Optional[float] = None,
) -> bool:
    if sent["pred"] != 2:
        return False
    probs = sent.get("probs") or []
    if class2_prob_threshold is not None:
        if len(probs) <= 2 or float(probs[2]) < class2_prob_threshold:
            return False
    if class2_margin_threshold is not None:
        if len(probs) <= 2:
            return False
        competitor = max(float(probs[0]), float(probs[1]))
        if float(probs[2]) - competitor < class2_margin_threshold:
            return False
    return True


def _first_class2_run(
    scored_sentences: List[dict],
    k: int,
    class2_prob_threshold: Optional[float] = None,
    class2_margin_threshold: Optional[float] = None,
) -> Optional[int]:
    if k <= 1:
        for idx, sent in enumerate(scored_sentences):
            if _passes_class2_gate(sent, class2_prob_threshold, class2_margin_threshold):
                return idx
        return None
    for idx in range(0, len(scored_sentences) - k + 1):
        if all(
            _passes_class2_gate(s, class2_prob_threshold, class2_margin_threshold)
            for s in scored_sentences[idx:idx + k]
        ):
            return idx + k - 1
    return None


def _run_early_exit(
    ctx,
    lm,
    tokenizer,
    scored_sentences: List[dict],
    k: int,
    max_new_tokens: int,
    device: str,
    is_code: bool,
    full_result: dict,
    class2_prob_threshold: Optional[float] = None,
    class2_margin_threshold: Optional[float] = None,
) -> dict:
    t0 = time.time()
    n_total = ctx.cot_end - ctx.cot_start
    experiment = _early_exit_name(k, class2_prob_threshold, class2_margin_threshold)
    exit_idx = _first_class2_run(
        scored_sentences,
        k,
        class2_prob_threshold=class2_prob_threshold,
        class2_margin_threshold=class2_margin_threshold,
    )
    if exit_idx is None:
        result = dict(full_result)
        result.update({
            "experiment": experiment,
            "k": k,
            "class2_prob_threshold": class2_prob_threshold,
            "class2_margin_threshold": class2_margin_threshold,
            "fallback_full": True,
            "exit_sentence_idx": None,
            "exit_sentence_frac": None,
            "trigger_sentence": None,
            "seconds": time.time() - t0,
        })
        return result

    sent = scored_sentences[exit_idx]
    exit_end_pos = min(int(sent["end_pos"]), ctx.cot_end - ctx.n_end_think)
    end_think = list(ctx.full_ids[ctx.cot_end - ctx.n_end_think:ctx.cot_end])
    suffix = list(ctx.full_ids[ctx.suffix_start:])
    ids = list(ctx.full_ids[:exit_end_pos]) + end_think + suffix
    generated, n_gen = _generate(lm, tokenizer, ids, max_new_tokens, device)
    correct = None if is_code else check_correct(generated, ctx.gt_answer)
    n_kept = max(0, exit_end_pos - ctx.cot_start) + ctx.n_end_think
    saved = max(0, n_total - n_kept)
    return {
        "experiment": experiment,
        "k": k,
        "class2_prob_threshold": class2_prob_threshold,
        "class2_margin_threshold": class2_margin_threshold,
        "correct": correct,
        "generated": generated,
        "n_gen_tokens": n_gen,
        "n_cot_tokens_kept": n_kept,
        "n_cot_tokens_total": n_total,
        "tokens_saved": saved,
        "saved_fraction": saved / max(n_total, 1),
        "fallback_full": False,
        "exit_sentence_idx": int(sent["sentence_index"]),
        "exit_sentence_frac": (exit_idx + 1) / max(len(scored_sentences), 1),
        "trigger_sentence": sent,
        "seconds": time.time() - t0,
    }


def _boundary_metrics_from_results(
    trace_records: List[dict],
    early_exit_variants: List[Tuple[str, int, Optional[float], Optional[float]]],
) -> dict:
    out = {}
    for name, k, class2_prob_threshold, class2_margin_threshold in early_exit_variants:
        n_total = len(trace_records)
        n_positive = 0
        n_negative = 0
        early = 0
        miss = 0
        detect = 0
        exact = 0
        false_negative_trigger = 0
        delays = []
        delay_fractions = []
        saved_all = []
        for rec in trace_records:
            scored = rec.get("scored_sentences") or []
            labels = [s["label"] for s in scored]
            true_indices = [i for i, label in enumerate(labels) if label == 2]
            pred = rec["experiments"].get(name, {})
            pred_sentence_idx = pred.get("exit_sentence_idx")
            pred_local_idx = None
            if pred_sentence_idx is not None:
                for i, sent in enumerate(scored):
                    if sent["sentence_index"] == pred_sentence_idx:
                        pred_local_idx = i
                        break
            if not true_indices:
                n_negative += 1
                if pred_local_idx is not None:
                    false_negative_trigger += 1
                continue
            n_positive += 1
            true_start = true_indices[0]
            if pred_local_idx is None:
                miss += 1
                saved_all.append(0.0)
                continue
            saved_all.append(float(pred.get("saved_fraction", 0.0)))
            if pred_local_idx < true_start:
                early += 1
                continue
            delay = pred_local_idx - true_start
            delays.append(delay)
            delay_fractions.append(delay / max(len(scored), 1))
            detect += 1
            exact += int(delay == 0)
        out[name] = {
            "n_traces": n_total,
            "experiment": name,
            "k": k,
            "class2_prob_threshold": class2_prob_threshold,
            "class2_margin_threshold": class2_margin_threshold,
            "n_positive_traces": n_positive,
            "n_negative_traces": n_negative,
            "positive_trace_rate": n_positive / n_total if n_total else 0.0,
            "positive_trace_rate_ci": _wilson_ci(n_positive, n_total),
            "negative_trace_rate": n_negative / n_total if n_total else 0.0,
            "negative_trace_rate_ci": _wilson_ci(n_negative, n_total),
            "detect_rate": detect / n_positive if n_positive else 0.0,
            "detect_rate_ci": _wilson_ci(detect, n_positive),
            "early_fire_rate": early / n_positive if n_positive else 0.0,
            "early_fire_rate_ci": _wilson_ci(early, n_positive),
            "miss_rate": miss / n_positive if n_positive else 0.0,
            "miss_rate_ci": _wilson_ci(miss, n_positive),
            "exact_boundary_rate": exact / n_positive if n_positive else 0.0,
            "exact_boundary_rate_ci": _wilson_ci(exact, n_positive),
            "false_trigger_rate_negative": (
                false_negative_trigger / n_negative if n_negative else None
            ),
            "false_trigger_rate_negative_ci": _wilson_ci(false_negative_trigger, n_negative),
            "mean_delay": sum(delays) / len(delays) if delays else None,
            "mean_delay_ci": _bootstrap_ci(delays, lambda vals: sum(vals) / len(vals)),
            "median_delay": float(statistics.median(delays)) if delays else None,
            "median_delay_ci": _bootstrap_ci(delays, lambda vals: float(statistics.median(vals))),
            "mean_delay_fraction": (
                sum(delay_fractions) / len(delay_fractions) if delay_fractions else None
            ),
            "mean_delay_fraction_ci": _bootstrap_ci(
                delay_fractions, lambda vals: sum(vals) / len(vals)
            ),
            "median_delay_fraction": (
                float(statistics.median(delay_fractions)) if delay_fractions else None
            ),
            "median_delay_fraction_ci": _bootstrap_ci(
                delay_fractions, lambda vals: float(statistics.median(vals))
            ),
            "mean_saved_fraction_all": sum(saved_all) / len(saved_all) if saved_all else 0.0,
            "mean_saved_fraction_all_ci": _bootstrap_ci(
                saved_all, lambda vals: sum(vals) / len(vals)
            ),
        }
    return out


def _pareto_front(experiment_summaries: Dict[str, dict]) -> List[dict]:
    points = []
    for name, summary in experiment_summaries.items():
        if summary.get("n_total", 0) <= 0:
            continue
        points.append({
            "experiment": name,
            "accuracy": float(summary.get("accuracy", 0.0)),
            "mean_saved_fraction": float(summary.get("mean_saved_fraction") or 0.0),
            "n_total": int(summary.get("n_total", 0)),
        })

    front = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if (
                other["accuracy"] >= point["accuracy"]
                and other["mean_saved_fraction"] >= point["mean_saved_fraction"]
                and (
                    other["accuracy"] > point["accuracy"]
                    or other["mean_saved_fraction"] > point["mean_saved_fraction"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda x: (x["mean_saved_fraction"], x["accuracy"]))


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main(
    model: str,
    data_name: str,
    probe_dir: str,
    probe_layer: int,
    attr_dir: Optional[str] = None,
    attribution_glob: str = "question_*.json",
    eval_question_subset: str = "auto",
    question_ids: Optional[str] = None,
    trace_indices: Optional[str] = None,
    exit_ks: str = "1,2,5,10",
    fixed_exit_percentages: str = "50,70,80,90,95",
    max_new_tokens: int = 10,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    lm_device_map: Optional[str] = None,
    seed: int = 42,
    split_file: Optional[str] = None,
    max_probe_input_tokens: Optional[int] = None,
    class2_prob_thresholds: Optional[str] = None,
    class2_margin_thresholds: Optional[str] = None,
    max_questions: Optional[int] = None,
    max_traces_per_question: Optional[int] = None,
):
    max_probe_input_tokens = _normalize_optional_positive_int(
        max_probe_input_tokens,
        "max_probe_input_tokens",
    )
    torch.manual_seed(seed)

    probe_path = Path(probe_dir).resolve()
    attr_path = Path(attr_dir).resolve() if attr_dir else _default_attr_dir(model, data_name).resolve()
    k_values = _parse_int_list(exit_ks) or [1, 2, 5, 10]
    class2_prob_values = [None] + (_parse_float_list(class2_prob_thresholds) or [])
    class2_margin_values = [None] + (_parse_float_list(class2_margin_thresholds) or [])
    class2_prob_values = list(dict.fromkeys(class2_prob_values))
    class2_margin_values = list(dict.fromkeys(class2_margin_values))
    early_exit_variants = [
        (_early_exit_name(k, p2, margin), k, p2, margin)
        for k in k_values
        for p2 in class2_prob_values
        for margin in class2_margin_values
    ]
    fixed_percentages = _parse_float_list(fixed_exit_percentages) or []
    for percent in fixed_percentages:
        if not (0.0 <= percent <= 100.0):
            raise ValueError(f"fixed_exit_percentages values must be in [0, 100], got {percent}")
    requested_question_ids = _parse_int_list(question_ids)
    requested_trace_indices = _parse_int_list(trace_indices)

    probe_results, probe_results_path, probe_weights_path = _load_probe_results(probe_path, int(probe_layer))
    probe_context_policy = probe_results.get("context_policy")
    if probe_context_policy != CONTEXT_POLICY:
        raise ValueError(
            f"Probe results at {probe_results_path} have context_policy={probe_context_policy!r}; "
            f"expected {CONTEXT_POLICY!r}. Retrain the probe so max_probe_input_tokens is a "
            "sliding causal lookback window instead of a tail truncation."
        )
    hparams = probe_results.get("hyperparams", {})
    sentence_aggregation = hparams.get("sentence_aggregation", "last")
    if sentence_aggregation not in {"last", "avg", "full"}:
        raise ValueError(f"Unsupported sentence_aggregation={sentence_aggregation!r}")
    max_sentences = hparams.get("max_sentences")
    hparam_max_probe_input_tokens = hparams.get("max_probe_input_tokens")
    if hparam_max_probe_input_tokens is None:
        hparam_max_probe_input_tokens = hparams.get("max_seq_tokens")
    hparam_max_probe_input_tokens = _normalize_optional_positive_int(
        hparam_max_probe_input_tokens,
        "probe hyperparam max_probe_input_tokens",
    )
    effective_max_probe_input_tokens = (
        max_probe_input_tokens if max_probe_input_tokens is not None else hparam_max_probe_input_tokens
    )

    print(f"Loading three-way solution probe from {probe_path}")
    print(f"  results: {probe_results_path.name}")
    print(f"  weights: {probe_weights_path.name}")
    print(f"  sentence_aggregation={sentence_aggregation}")
    print(f"  context_policy={CONTEXT_POLICY}")
    print(f"  causal_lookback_window={effective_max_probe_input_tokens if effective_max_probe_input_tokens is not None else 'full prefix'}")
    print(f"  class2_prob_thresholds={class2_prob_values}")
    print(f"  class2_margin_thresholds={class2_margin_values}")
    print(f"Attribution dir: {attr_path}")

    trace_labels = load_solution_attribution_data(
        str(attr_path),
        attribution_glob=attribution_glob,
    )
    traces_data = get_reasoning_traces(model, data_name)
    print(f"Loaded {len(traces_data)} questions from trace dataset")

    split, split_path = _load_split(probe_path, attr_path, split_file)
    if split_path:
        print(f"Question split loaded from: {split_path}")
    else:
        print("No question split found; eval_question_subset=auto/all will use all questions")

    train_data_name = probe_results.get("data_name", data_name)
    selected_questions = _select_question_ids(
        len(traces_data), split, eval_question_subset, train_data_name, data_name
    )
    if requested_question_ids is not None:
        allowed = set(selected_questions)
        selected_questions = [q for q in requested_question_ids if q in allowed]
    if max_questions is not None:
        selected_questions = selected_questions[:max_questions]
    if not selected_questions:
        raise ValueError("No questions selected for evaluation")

    is_code = is_code_benchmark(data_name)
    is_live_code = "live_code_bench" in data_name.casefold()
    if is_code and max_new_tokens == 10:
        max_new_tokens = 1500
        print(f"Auto-increased max_new_tokens to {max_new_tokens} for code benchmark")
    code_meta = get_code_benchmark_metadata(data_name)

    print(f"Loading model: {model}")
    lm = load_nnsight_model(model, device=device, device_map=lm_device_map)
    tokenizer = lm.tokenizer
    cfg = detect_model_config(lm)
    print(f"Model: {cfg.n_layers} layers, d_model={cfg.d_model}")

    layer = int(probe_layer)
    if layer < 0:
        layer = cfg.n_layers + layer
    if not (0 <= layer < cfg.n_layers):
        raise ValueError(f"probe_layer={probe_layer} resolves to invalid layer {layer}")

    probe = SentenceCausalProbe(
        cfg.d_model,
        hidden_dim=int(hparams.get("hidden_dim", 64)),
        dropout=float(hparams.get("dropout", 0.1)),
        n_classes=3,
    )
    state = torch.load(probe_weights_path, map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.to(device)
    probe.eval()

    thinking_tokens = get_thinking_tokens(model)
    start_ids = thinking_tokens.get("start_token_ids")
    if start_ids is None:
        start_ids = tokenizer.encode(thinking_tokens["start_token"], add_special_tokens=False)
    end_ids = thinking_tokens.get("end_token_ids")
    if end_ids is None:
        end_ids = tokenizer.encode(thinking_tokens["end_token"], add_special_tokens=False)
    answer_suffix = get_answer_suffix(data_name)
    suffix_ids = tokenizer.encode(answer_suffix, add_special_tokens=False)

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = probe_path / f"early_exit_data-{data_name.split('/')[-1]}_L{layer}"
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model,
        "data_name": data_name,
        "probe_dir": str(probe_path),
        "probe_results": str(probe_results_path),
        "probe_weights": str(probe_weights_path),
        "attr_dir": str(attr_path),
        "probe_layer": layer,
        "task_mode": "three_way",
        "context_policy": CONTEXT_POLICY,
        "sentence_aggregation": sentence_aggregation,
        "probe_hparam_max_probe_input_tokens": hparam_max_probe_input_tokens,
        "max_probe_input_tokens": effective_max_probe_input_tokens,
        "exit_ks": k_values,
        "class2_prob_thresholds": class2_prob_values,
        "class2_margin_thresholds": class2_margin_values,
        "fixed_exit_percentages": fixed_percentages,
        "eval_question_subset": eval_question_subset,
        "question_split": str(split_path) if split_path else None,
        "n_eval_questions": len(selected_questions),
        "max_new_tokens": max_new_tokens,
    }

    accumulators: Dict[str, VariantAccumulator] = {
        "baseline_full": VariantAccumulator("baseline_full"),
        "baseline_nocot": VariantAccumulator("baseline_nocot"),
    }
    for percent in fixed_percentages:
        name = f"baseline_fixed_p{_percentage_tag(percent)}"
        accumulators[name] = VariantAccumulator(name)
    for name, _k, _p2, _margin in early_exit_variants:
        accumulators[name] = VariantAccumulator(name)

    all_trace_records: List[dict] = []
    total_traces = 0
    for q_id in selected_questions:
        if q_id >= len(traces_data):
            continue
        n_traces = len(traces_data[q_id]["traces_tokens"])
        active = requested_trace_indices if requested_trace_indices is not None else list(range(n_traces))
        if max_traces_per_question is not None:
            active = active[:max_traces_per_question]
        total_traces += len(active)

    print(f"Will evaluate {len(selected_questions)} questions / {total_traces} traces")
    print(f"Outputs: {out_path}")

    t_start = time.time()
    traces_done = 0

    def save_summary() -> None:
        summaries = {name: acc.summary() for name, acc in accumulators.items()}
        summary = {
            "metadata": {**metadata, "timestamp": datetime.now().isoformat()},
            "elapsed_seconds": time.time() - t_start,
            "experiments": summaries,
            "pareto_front": _pareto_front(summaries),
            "boundary_metrics": _boundary_metrics_from_results(all_trace_records, early_exit_variants),
        }
        _save_json(out_path / "summary.json", summary)

    for q_id in selected_questions:
        q_data = traces_data[q_id]
        gt_answer = str(q_data["GT_answer"])
        traces_tokens = q_data["traces_tokens"]
        active_trace_indices = (
            requested_trace_indices if requested_trace_indices is not None else list(range(len(traces_tokens)))
        )
        if max_traces_per_question is not None:
            active_trace_indices = active_trace_indices[:max_traces_per_question]

        q_task_id = q_data.get("task_id")
        q_entry_point = q_data.get("entry_point")
        q_tests = q_data.get("tests")
        if q_task_id is None and code_meta is not None and q_id < len(code_meta):
            q_task_id = code_meta[q_id]["task_id"]
            q_entry_point = code_meta[q_id].get("entry_point")
            q_tests = code_meta[q_id].get("tests", q_tests)

        q_trace_records = []
        code_pending = []
        sol_counter: Dict[str, int] = {}

        for t_idx in active_trace_indices:
            key = (q_id, t_idx)
            if key not in trace_labels:
                print(f"  Q{q_id} T{t_idx}: skipped (no sentence-causal labels)")
                traces_done += 1
                continue

            ctx = build_trace_context(
                q_data,
                q_id,
                t_idx,
                tokenizer,
                end_ids,
                suffix_ids,
                start_ids=start_ids,
                task_id=q_task_id,
                entry_point=q_entry_point,
            )
            if ctx is None:
                print(f"  Q{q_id} T{t_idx}: skipped (no end-thinking marker)")
                traces_done += 1
                continue

            n_total = ctx.cot_end - ctx.cot_start
            print(f"  Q{q_id} T{t_idx}: {n_total} CoT tokens")
            trace_t0 = time.time()

            try:
                full_result = _run_full_baseline(ctx, lm, tokenizer, max_new_tokens, device, is_code)
                nocot_result = _run_nocot_baseline(ctx, lm, tokenizer, max_new_tokens, device, is_code)

                full_ids_t = torch.tensor([ctx.full_ids], device=device)
                resid_t0 = time.time()
                full_resid = collect_residual_stream(lm, cfg, full_ids_t)
                scored_sentences = _score_sentences(
                    probe,
                    full_resid,
                    layer,
                    trace_labels[key],
                    sentence_aggregation,
                    device,
                    max_sentences=max_sentences,
                    max_probe_input_tokens=effective_max_probe_input_tokens,
                )
                del full_resid
                resid_seconds = time.time() - resid_t0

                experiments = {
                    "baseline_full": full_result,
                    "baseline_nocot": nocot_result,
                }
                for percent in fixed_percentages:
                    result = _run_fixed_percentage_exit(
                        ctx,
                        lm,
                        tokenizer,
                        trace_labels[key],
                        percent,
                        max_new_tokens,
                        device,
                        is_code,
                        full_result,
                        nocot_result,
                    )
                    experiments[result["experiment"]] = result
                for name, k, p2_threshold, margin_threshold in early_exit_variants:
                    result = _run_early_exit(
                        ctx,
                        lm,
                        tokenizer,
                        scored_sentences,
                        k,
                        max_new_tokens,
                        device,
                        is_code,
                        full_result,
                        class2_prob_threshold=p2_threshold,
                        class2_margin_threshold=margin_threshold,
                    )
                    experiments[name] = result

                if not is_code:
                    for name, result in experiments.items():
                        accumulators[name].add(result)

                if is_code and q_task_id:
                    for name, result in experiments.items():
                        generated = result.get("generated", "")
                        if not generated:
                            continue
                        sol_idx = sol_counter.get(q_task_id, 0)
                        code_pending.append({
                            "experiment": name,
                            "task_id": q_task_id,
                            "solution": assemble_solution_for_eval(answer_suffix, generated),
                            "result_ref": result,
                            "sol_idx": sol_idx,
                        })
                        sol_counter[q_task_id] = sol_idx + 1

                trace_record = {
                    "trace_index": t_idx,
                    "n_tokens_full": len(ctx.full_ids),
                    "n_cot_tokens": n_total,
                    "gt_answer": gt_answer,
                    "residual_collection_seconds": resid_seconds,
                    "scored_sentences": scored_sentences,
                    "experiments": experiments,
                    "seconds": time.time() - trace_t0,
                }
                q_trace_records.append(trace_record)
                all_trace_records.append(trace_record)

                row = []
                display_names = (
                    ["baseline_full", "baseline_nocot"]
                    + [f"baseline_fixed_p{_percentage_tag(p)}" for p in fixed_percentages]
                    + [name for name, _k, _p2, _margin in early_exit_variants]
                )
                for name in display_names:
                    result = experiments[name]
                    correct = "code" if is_code else ("Y" if result.get("correct") else "N")
                    saved = 100.0 * float(result.get("saved_fraction", 0.0))
                    exit_s = result.get("exit_sentence_idx")
                    exit_txt = "full" if result.get("fallback_full") else ("-" if exit_s is None else str(exit_s))
                    row.append(f"{name}: {correct}, saved={saved:.1f}%, exit={exit_txt}")
                print("    " + " | ".join(row))

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                    print(f"    skipped (OOM)")
                else:
                    raise
            finally:
                traces_done += 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elapsed = time.time() - t_start
            avg = elapsed / max(traces_done, 1)
            remaining = avg * max(total_traces - traces_done, 0)
            print(f"    done [{traces_done}/{total_traces}], ~{remaining / 60:.1f}m remaining")

        if is_code and code_pending:
            eval_label = "live_code_bench" if is_live_code else "EvalPlus"
            print(f"  Q{q_id}: running {eval_label} on {len(code_pending)} generations")
            eval_samples = [{"task_id": p["task_id"], "solution": p["solution"]} for p in code_pending]
            eval_results = run_code_eval(
                eval_samples,
                out_dir=out_path / ("live_code_bench_tmp" if is_live_code else "evalplus_tmp") / f"q{q_id:04d}",
                is_live_code=is_live_code,
                tests_by_task=({q_task_id: q_tests or []} if is_live_code else None),
            )
            for pending in code_pending:
                task_entries = eval_results.get(pending["task_id"], [])
                result_ref = pending["result_ref"]
                sol_idx = pending.get("sol_idx", 0)
                if sol_idx < len(task_entries):
                    is_pass, fail_reason = parse_code_eval_entry(task_entries[sol_idx])
                else:
                    is_pass, fail_reason = False, "no eval entry"
                result_ref["correct"] = is_pass
                result_ref["fail_reason"] = fail_reason
            for rec in q_trace_records:
                for name, result in rec["experiments"].items():
                    accumulators[name].add(result)

        q_json = {
            "metadata": metadata,
            "question_id": q_id,
            "GT_answer": gt_answer,
            "traces": q_trace_records,
        }
        _save_json(out_path / f"early_exit_q{q_id:04d}.json", q_json)
        save_summary()
        print(f"  Q{q_id}: saved {len(q_trace_records)} traces")

    save_summary()

    summaries = {name: acc.summary() for name, acc in accumulators.items()}
    print()
    print("=" * 80)
    print("EARLY-EXIT SUMMARY")
    print("=" * 80)
    print(f"{'experiment':<30} {'accuracy':>10} {'saved':>10} {'fallback':>10} {'N':>6}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")
    for name, summary in summaries.items():
        saved = summary.get("mean_saved_fraction")
        fallback = summary.get("fallback_full_rate")
        print(
            f"{name:<30} {summary['accuracy']:>10.3f} "
            f"{(100.0 * saved if saved is not None else 0.0):>9.1f}% "
            f"{(100.0 * fallback if fallback is not None else 0.0):>9.1f}% "
            f"{summary['n_total']:>6}"
        )
    print()
    print("Pareto front (accuracy vs mean saved fraction):")
    for point in _pareto_front(summaries):
        print(
            f"  {point['experiment']:<30} acc={point['accuracy']:.3f} "
            f"saved={100.0 * point['mean_saved_fraction']:.1f}% N={point['n_total']}"
        )
    print(f"\nSummary written to {out_path / 'summary.json'}")


if __name__ == "__main__":
    fire.Fire(main)
