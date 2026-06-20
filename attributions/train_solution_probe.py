#!/usr/bin/env python3
"""
Train a simple sentence-state causal probe to detect whether the model has a
plausible solution in mind at each sentence boundary.

Unlike train_attn_probe.py, this script does not try to localize a target-token
spike. Sentence-causal JSONs provide semantic guess labels directly; the
attribution pipeline is responsible for applying any confidence gate before
labels reach this trainer.

For semantic attribution files, three-way labels are:
    0 = no usable or not-yet-confident guess
    1 = confident wrong/mid guess
    2 = confident final-equivalent guess

Binary labels are 1 for semantic labels 1 or 2.

The probe operates over one state per sentence, with causal attention over the
sentence prefix. Default aggregation is the last token of each sentence.
"""

import gc
import json
import re
import sys
from glob import glob as file_glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import torch
import torch.nn as nn
from tqdm import tqdm

from attributions.utils import get_reasoning_traces, generate_question_split
from attributions.shared import (
    compute_classification_metrics as _compute_classification_metrics,
    compute_random_baselines as _compute_random_baselines,
    print_baselines as _print_baselines,
    bucket_files_by_question as _bucket_files_by_question,
    sweep_threshold_curve,
)
from attributions.activation_cache import collect_and_cache
from attributions.modeling import detect_model_config, load_nnsight_model

SEMANTIC_LABEL_POLICY = "semantic_equivalence_gated_by_first_token_confidence"
CONTEXT_POLICY = "sliding_causal_window"


def _default_cache_dir(model_short: str, data_short: str) -> Path:
    """Default to the shared probe cache used by the other probe scripts."""
    return Path("outputs") / model_short / data_short / "probe_cache"


def _get_span_semantic_label(span: dict) -> Optional[int]:
    value = span.get("semantic_label")
    if value is None:
        value = span.get("semantic_guess", {}).get("semantic_label")
    if value is None:
        return None
    label = int(value)
    if label not in (0, 1, 2):
        raise ValueError(f"Invalid semantic_label={label}; expected one of 0, 1, 2")
    return label


class SentenceCausalProbe(nn.Module):
    """Causal attention probe over sentence states."""

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.1, n_classes: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, 1, bias=False)
        self.W_v = nn.Linear(d_model, hidden_dim, bias=False)
        self.W_local = nn.Linear(d_model, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.n_classes = n_classes
        self.W_out = nn.Linear(2 * hidden_dim, n_classes)

    @staticmethod
    def causal_attention_mask(
        seq_len: int,
        device: torch.device,
        causal_window: Optional[int] = None,
    ) -> torch.Tensor:
        if causal_window is not None and causal_window <= 0:
            raise ValueError(f"causal_window must be positive or None, got {causal_window}")
        positions = torch.arange(seq_len, device=device)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        if causal_window is None:
            return causal
        return causal & (positions.unsqueeze(1) - positions.unsqueeze(0) < causal_window)

    def forward(
        self,
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal_window: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, _ = H.shape
        device = H.device

        H = self.norm(H)
        local = self.W_local(H)
        scores = self.W_q(H).squeeze(-1)
        values = self.W_v(H)

        causal = self.causal_attention_mask(N, device, causal_window=causal_window)
        scores_expanded = scores.unsqueeze(1).expand(B, N, N)
        attn_mask = causal.unsqueeze(0)
        if mask is not None:
            attn_mask = attn_mask & mask.unsqueeze(1)

        masked_scores = scores_expanded.masked_fill(~attn_mask, float("-inf"))
        alpha = torch.softmax(masked_scores, dim=-1).nan_to_num(0.0)
        context = torch.bmm(alpha, values)
        combined = torch.cat([context, local], dim=-1)
        out = self.W_out(self.drop(torch.relu(combined)))
        return out.squeeze(-1) if self.n_classes == 1 else out


def load_solution_attribution_data(
    attr_dir: str,
    attribution_glob: str = "question_*.json",
    clue_alpha: Optional[float] = None,
) -> Dict[Tuple[int, int], dict]:
    """Load sentence-causal JSONs and consume emitted semantic labels."""
    if clue_alpha is not None:
        print(
            "WARNING: load_solution_attribution_data ignores clue_alpha; "
            "apply confidence gating when running sentence-causal attribution."
        )
    pattern = str(Path(attr_dir) / attribution_glob)
    files = sorted(file_glob(pattern))
    files = [f for f in files if not f.endswith("_eval.json") and not f.endswith("_split.json")]
    if not files:
        raise FileNotFoundError(f"No attribution files found matching {pattern}")

    labels: Dict[Tuple[int, int], dict] = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        q_id = data["question_id"]
        for trace in data.get("traces", []):
            t_idx = trace["trace_index"]
            key = (q_id, t_idx)
            if key in labels or trace.get("skipped"):
                continue

            sentence_spans = trace.get("sentence_spans")
            full_ids = trace.get("full_ids")
            target_pos = trace.get("target_pos")
            p_no = trace.get("target_token_prob_no_cot")
            p_full = trace.get("target_token_prob_full_cot")
            if sentence_spans is None or full_ids is None or target_pos is None:
                continue

            clue_labels = []
            three_way_labels = []
            actual_token_probs = []
            target_token_id = trace.get("target_token_id")
            if target_token_id is None:
                continue

            semantic_meta = trace.get("semantic_guess_labels")
            semantic_labels = [_get_span_semantic_label(span) for span in sentence_spans]
            has_semantic_labels = (
                isinstance(semantic_meta, dict)
                and bool(semantic_meta.get("enabled"))
                and all(label is not None for label in semantic_labels)
            )

            if not has_semantic_labels:
                raise ValueError(
                    f"Trace (q{q_id}, t{t_idx}) in {fpath} is missing enabled semantic labels. "
                    "Rerun sentence-causal attribution with --semantic_guess_labels True and --clue_alpha."
                )
            format_version = int(semantic_meta.get("format_version") or 0)
            threshold = semantic_meta.get("clue_alpha_threshold")
            if (
                format_version < 2
                or semantic_meta.get("label_policy") != SEMANTIC_LABEL_POLICY
                or threshold is None
            ):
                raise ValueError(
                    f"Trace (q{q_id}, t{t_idx}) in {fpath} has stale or ungated semantic labels. "
                    "Rerun sentence-causal attribution so semantic_guess_labels records "
                    f"format_version>=2, label_policy={SEMANTIC_LABEL_POLICY!r}, and clue_alpha_threshold."
                )

            threshold = float(threshold)
            for span, semantic_label in zip(sentence_spans, semantic_labels):
                assert semantic_label is not None
                three_way_labels.append(int(semantic_label))
                clue_labels.append(int(semantic_label != 0))
                semantic_guess = span.get("semantic_guess", {})
                p_act = semantic_guess.get("first_token_prob")
                if p_act is None:
                    p_act = span.get("actual_predicted_token", {}).get("token_prob", 0.0)
                actual_token_probs.append(float(p_act))
            label_source = "semantic_guess_labels"

            sentence_scores = trace.get("sentence_importance_scores")
            if sentence_scores is None:
                sentence_scores = [0.0] * len(sentence_spans)

            labels[key] = {
                "full_ids": full_ids,
                "target_pos": target_pos,
                "n_tokens": trace.get("n_tokens", len(full_ids)),
                "granularity": "sentence",
                "unit_scores": sentence_scores,
                "unit_spans": sentence_spans,
                "clue_labels": clue_labels,
                "three_way_labels": three_way_labels,
                "actual_token_probs": actual_token_probs,
                "clue_threshold": threshold,
                "label_source": label_source,
                "semantic_label_policy": semantic_meta.get("label_policy"),
                "semantic_guess_labels": semantic_meta,
                "target_token_id": int(target_token_id),
                "target_token_prob_no_cot": float(p_no) if p_no is not None else None,
                "target_token_prob_full_cot": float(p_full) if p_full is not None else None,
            }

    print(f"Loaded {len(labels)} traces from {len(files)} attribution files")
    return labels


def _parse_cache_ids(filename: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"q(\d+)_t(\d+)_L\d+\.pt", Path(filename).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _normalize_optional_positive_int(value: object, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.casefold() in {"none", "null", ""}:
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive or None, got {value!r}")
    return ivalue


def _check_cache_complete_for_layer(cache_dir: Path, labels: dict, layer: int) -> Tuple[bool, str]:
    """Check whether every labeled trace has a cache file for one resolved layer."""
    if not cache_dir.exists():
        return False, "cache directory does not exist"
    missing = [
        key for key in labels
        if not (cache_dir / f"q{key[0]}_t{key[1]}_L{layer}.pt").exists()
    ]
    if not missing:
        n_extra = len(list(cache_dir.glob(f"*_L{layer}.pt"))) - len(labels)
        extra_str = f", {n_extra} extra (OOM skips or eval traces)" if n_extra > 0 else ""
        return True, f"all {len(labels)} traces cached for L{layer}{extra_str}"
    return False, f"{len(missing)}/{len(labels)} traces missing from cache for L{layer}"


def _aggregate_sentence_states(
    acts: torch.Tensor,
    spans: List[List[int]],
    aggregation: str,
) -> Tuple[torch.Tensor, List[int]]:
    states = []
    kept_indices = []
    for idx, span in enumerate(spans):
        start, end = int(span[0]), int(span[1])
        if start >= end or start < 0 or end > len(acts):
            continue
        if aggregation == "last":
            state = acts[end - 1]
        elif aggregation == "avg":
            state = acts[start:end].mean(dim=0)
        else:
            raise ValueError(f"Unknown sentence_aggregation: {aggregation}")
        states.append(state)
        kept_indices.append(idx)
    if not states:
        return torch.empty(0, acts.shape[-1]), []
    return torch.stack(states), kept_indices


def _build_full_sentence_sequence(
    acts: torch.Tensor,
    spans: List[List[int]],
    sentence_labels: List[int],
    sentence_end_positions: List[int],
    sentence_start_positions: List[int],
    cot_end_pos: int,
    max_sentences: Optional[int] = None,
    max_probe_input_tokens: Optional[int] = None,
) -> Optional[dict]:
    kept_unit_indices = list(range(len(spans)))
    if max_sentences is not None and len(kept_unit_indices) > max_sentences:
        kept_unit_indices = kept_unit_indices[-max_sentences:]

    kept_spans = [spans[idx] for idx in kept_unit_indices]
    if not kept_spans:
        return None

    start_offset = int(kept_spans[0][0])
    end_offset = int(kept_spans[-1][1])
    if start_offset >= end_offset:
        return None
    if max_probe_input_tokens is not None and end_offset - start_offset > max_probe_input_tokens:
        start_offset = max(start_offset, end_offset - max_probe_input_tokens)

    acts = acts[start_offset:end_offset]
    labels = torch.zeros(len(acts), dtype=torch.long)
    loss_mask = torch.zeros(len(acts), dtype=torch.bool)
    kept_end_positions = []
    kept_sentence_start_positions = []

    for local_idx, unit_idx in enumerate(kept_unit_indices):
        _start, end = kept_spans[local_idx]
        end_idx = int(end - 1 - start_offset)
        if end_idx < 0 or end_idx >= len(acts):
            continue
        labels[end_idx] = int(sentence_labels[unit_idx])
        loss_mask[end_idx] = True
        kept_end_positions.append(int(sentence_end_positions[unit_idx]))
        kept_sentence_start_positions.append(int(sentence_start_positions[unit_idx]))

    if not loss_mask.any():
        return None

    return {
        "activations": acts,
        "labels": labels,
        "loss_mask": loss_mask,
        "sentence_end_positions": kept_end_positions,
        "cot_start_pos": int(kept_sentence_start_positions[0]),
        "cot_end_pos": cot_end_pos,
    }


def _spans_from_trace_labels_and_positions(
    trace_info: dict,
    positions: Optional[List[int]],
) -> Tuple[List[List[int]], List[int]]:
    """Reconstruct CoT-local sentence spans from attribution JSON spans.

    Args:
        trace_info: Entry from load_solution_attribution_data().
        positions: Full-trace token positions stored in cache for each row of
            the activation tensor.

    Returns:
        A pair of reconstructed local spans and the original JSON span indices
        those spans came from.
    """
    raw_spans = trace_info.get("unit_spans") or []
    if not raw_spans:
        return [], []
    if positions is None:
        raise ValueError(
            "Cache file is missing both unit_spans and positions, so sentence "
            "boundaries cannot be reconstructed from the attribution JSON."
        )

    pos_to_local = {int(pos): idx for idx, pos in enumerate(positions)}
    spans_local: List[List[int]] = []
    span_label_indices: List[int] = []
    for raw_idx, span in enumerate(raw_spans):
        start_full = int(span["start_pos"])
        end_full = int(span["end_pos"])
        local_indices = [pos_to_local[pos] for pos in range(start_full, end_full) if pos in pos_to_local]
        if not local_indices:
            continue
        spans_local.append([local_indices[0], local_indices[-1] + 1])
        span_label_indices.append(raw_idx)
    return spans_local, span_label_indices


def prepare_solution_datasets(
    cache_dir: Path,
    trace_labels: Dict[Tuple[int, int], dict],
    layer: int,
    train_question_ids: List[int],
    val_question_ids: List[int],
    test_question_ids: List[int],
    seed: int = 42,
    holdout_trace_ratio: float = 0.15,
    sentence_aggregation: str = "last",
    task_mode: str = "binary",
    max_sentences: Optional[int] = None,
    max_probe_input_tokens: Optional[int] = None,
    max_traces_per_question: Optional[int] = None,
) -> Tuple[list, list, list, dict]:
    assert sentence_aggregation in ("full", "last", "avg"), (
        f"sentence_aggregation must be full|last|avg, got {sentence_aggregation}"
    )

    files = sorted(cache_dir.glob(f"*_L{layer}.pt"))
    if not files:
        raise FileNotFoundError(f"No cached files found for layer {layer} in {cache_dir}")

    buckets, skipped = _bucket_files_by_question(
        files,
        train_question_ids, val_question_ids, test_question_ids,
        seed=seed,
        max_traces_per_question=max_traces_per_question,
    )

    n_train_files = len(buckets["train"])
    n_val_files = len(buckets["val"])
    n_test_files = len(buckets["test"])
    n_total_files = n_train_files + n_val_files + n_test_files
    print(f"Question-level split: {n_train_files} train / {n_val_files} val / "
          f"{n_test_files} test traces ({n_total_files} total, "
          f"{len(files) - n_total_files - skipped} eval traces excluded)")

    holdout_files: List[str] = []
    if holdout_trace_ratio > 0 and n_train_files > 0:
        rng_traces = torch.Generator().manual_seed(seed)
        n_holdout = max(1, int(n_train_files * holdout_trace_ratio))
        trace_perm = torch.randperm(n_train_files, generator=rng_traces)
        holdout_indices = set(trace_perm[:n_holdout].tolist())
        holdout_files = [str(buckets["train"][i]) for i in sorted(holdout_indices)]
        buckets["train"] = [f for i, f in enumerate(buckets["train"]) if i not in holdout_indices]
        print(f"Held out {len(holdout_files)}/{n_train_files} train traces for per-trace eval")

    label_key = "clue_labels" if task_mode == "binary" else "three_way_labels"

    def _load_seqs(file_list: List[Path], desc: str) -> Tuple[list, Dict[int, int]]:
        seqs = []
        class_counts = {0: 0, 1: 0, 2: 0}
        for f in tqdm(file_list, desc=desc, unit="trace", leave=False):
            key = _parse_cache_ids(f.name)
            if key is None or key not in trace_labels:
                continue
            try:
                data = torch.load(f, weights_only=True)
            except (EOFError, RuntimeError, Exception) as e:
                tqdm.write(f"    WARNING: skipping corrupted cache file {f.name} ({type(e).__name__}: {e})")
                continue

            acts = data["activations"].float()
            spans = data.get("unit_spans")
            positions = data.get("positions")
            raw_sentence_spans = trace_labels[key]["unit_spans"]
            if spans is None:
                spans, span_label_indices = _spans_from_trace_labels_and_positions(trace_labels[key], positions)
            else:
                span_label_indices = list(range(len(spans)))
                if len(spans) != len(raw_sentence_spans):
                    raise ValueError(
                        f"Sentence-level cache file {f} has {len(spans)} unit_spans but attribution JSON "
                        f"has {len(raw_sentence_spans)} sentence_spans. Recollect the cache so labels "
                        "cannot be paired with the wrong sentence spans."
                    )
            if not spans:
                raise ValueError(
                    f"Sentence-level cache file {f} has no usable sentence spans. "
                    "Expected unit_spans in cache or reconstructable spans from cache positions "
                    "and attribution JSON sentence_spans."
                )
            cot_end_pos = int(raw_sentence_spans[-1]["end_pos"])
            sentence_start_positions = [
                int(raw_sentence_spans[idx]["start_pos"]) for idx in span_label_indices
            ]
            sentence_end_positions = [
                int(raw_sentence_spans[idx]["end_pos"]) for idx in span_label_indices
            ]
            sentence_labels = [
                trace_labels[key][label_key][idx] for idx in span_label_indices
            ]

            if sentence_aggregation == "full":
                seq = _build_full_sentence_sequence(
                    acts,
                    spans,
                    sentence_labels,
                    sentence_end_positions,
                    sentence_start_positions,
                    cot_end_pos,
                    max_sentences=max_sentences,
                    max_probe_input_tokens=max_probe_input_tokens,
                )
                if seq is None:
                    continue
                supervised_labels = seq["labels"][seq["loss_mask"]]
                labels_for_count = supervised_labels
            else:
                sent_acts, kept_indices = _aggregate_sentence_states(acts, spans, sentence_aggregation)
                if len(kept_indices) == 0:
                    continue

                labels = torch.tensor(
                    [sentence_labels[idx] for idx in kept_indices],
                    dtype=torch.long,
                )
                if max_sentences is not None and len(labels) > max_sentences:
                    sent_acts = sent_acts[-max_sentences:]
                    labels = labels[-max_sentences:]
                    kept_indices = kept_indices[-max_sentences:]

                seq = {
                    "activations": sent_acts,
                    "labels": labels,
                    "loss_mask": torch.ones(len(labels), dtype=torch.bool),
                    "sentence_end_positions": [sentence_end_positions[idx] for idx in kept_indices],
                    "cot_start_pos": sentence_start_positions[kept_indices[0]],
                    "cot_end_pos": cot_end_pos,
                }
                labels_for_count = labels

            seqs.append(seq)
            unique, counts = labels_for_count.unique(return_counts=True)
            for cls, count in zip(unique.tolist(), counts.tolist()):
                class_counts[int(cls)] = class_counts.get(int(cls), 0) + int(count)
        return seqs, class_counts

    train_seqs, train_counts = _load_seqs(buckets["train"], "Loading train")
    val_seqs, val_counts = _load_seqs(buckets["val"], "Loading val")
    test_seqs, test_counts = _load_seqs(buckets["test"], "Loading test")

    if not train_seqs:
        raise ValueError("No valid training sequences found")

    total_counts = {
        cls: train_counts.get(cls, 0) + val_counts.get(cls, 0) + test_counts.get(cls, 0)
        for cls in sorted(set(train_counts) | set(val_counts) | set(test_counts))
    }

    stats = {
        "task_mode": task_mode,
        "total_sentences": sum(total_counts.values()),
        "class_counts": total_counts,
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_test": len(test_seqs),
        "n_holdout_traces": len(holdout_files),
        "holdout_traces": holdout_files,
        "sentence_aggregation": sentence_aggregation,
        "max_sentences": max_sentences,
        "max_probe_input_tokens": max_probe_input_tokens,
    }
    if task_mode == "binary":
        train_pos = train_counts.get(1, 0)
        train_neg = train_counts.get(0, 0)
        stats["positive_sentences"] = total_counts.get(1, 0)
        stats["negative_sentences"] = total_counts.get(0, 0)
        if train_pos > 0:
            stats["pos_weight"] = train_neg / train_pos
        print(f"Dataset: {stats['total_sentences']} sentence states "
              f"({stats['positive_sentences']} pos, {stats['negative_sentences']} neg)")
    else:
        print(
            f"Dataset: {stats['total_sentences']} sentence states "
            f"(class0={total_counts.get(0, 0)}, class1={total_counts.get(1, 0)}, class2={total_counts.get(2, 0)})"
        )
    print(f"  Split: {stats['n_train']} train, {stats['n_val']} val, {stats['n_test']} test seqs")
    if "pos_weight" in stats:
        print(f"  pos_weight={stats['pos_weight']:.2f}")

    return train_seqs, val_seqs, test_seqs, stats


def collate_sequences(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_model = batch[0]["activations"].shape[1]
    lengths = [len(s["labels"]) for s in batch]
    max_len = max(lengths)
    B = len(batch)

    H = torch.zeros(B, max_len, d_model)
    y = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    loss_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, s in enumerate(batch):
        n = lengths[i]
        H[i, :n] = s["activations"]
        y[i, :n] = s["labels"]
        mask[i, :n] = True
        loss_mask[i, :n] = s.get("loss_mask", torch.ones(n, dtype=torch.bool))
    return H, y, mask, loss_mask


def _compute_multiclass_metrics(
    preds: torch.Tensor,
    y: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_classes: int = 3,
) -> dict:
    preds = preds.long()
    y = y.long()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_cls in range(num_classes):
        true_mask = y == true_cls
        for pred_cls in range(num_classes):
            confusion[true_cls, pred_cls] = ((preds == pred_cls) & true_mask).sum()

    n = int(confusion.sum().item())
    accuracy = confusion.diag().sum().item() / n if n > 0 else 0.0
    per_class = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for cls in range(num_classes):
        tp = int(confusion[cls, cls].item())
        fp = int(confusion[:, cls].sum().item() - tp)
        fn = int(confusion[cls, :].sum().item() - tp)
        support = int(confusion[cls, :].sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({
            "class_id": cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes

    auroc_macro_ovr = None
    if probs is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auroc_macro_ovr = float(roc_auc_score(y.cpu().numpy(), probs.cpu().numpy(), multi_class="ovr", average="macro"))
        except ImportError:
            print("WARNING: sklearn not found, multiclass AUROC not computed")
        except ValueError as e:
            print(f"WARNING: multiclass AUROC computation failed: {e}")

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "auroc_macro_ovr": auroc_macro_ovr,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "support": int(n),
    }


def _compute_multiclass_class_weights(train_seqs: list, num_classes: int = 3) -> Optional[torch.Tensor]:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for seq in train_seqs:
        loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
        labels = seq["labels"][loss_mask]
        if len(labels) == 0:
            continue
        unique, class_counts = labels.unique(return_counts=True)
        for cls, count in zip(unique.tolist(), class_counts.tolist()):
            if 0 <= int(cls) < num_classes:
                counts[int(cls)] += float(count)
    if (counts <= 0).any():
        return None
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights


def _first_index_of_class(labels: torch.Tensor, cls: int) -> Optional[int]:
    matches = (labels == cls).nonzero(as_tuple=False)
    if len(matches) == 0:
        return None
    return int(matches[0].item())


def _first_consecutive_run(preds: torch.Tensor, cls: int, run_len: int) -> Optional[int]:
    if run_len <= 1:
        return _first_index_of_class(preds, cls)
    if len(preds) < run_len:
        return None
    for idx in range(len(preds) - run_len + 1):
        if bool((preds[idx:idx + run_len] == cls).all().item()):
            return idx
    return None


def compute_class2_boundary_metrics(
    preds_by_seq: List[torch.Tensor],
    labels_by_seq: List[torch.Tensor],
    seqs: list,
    max_k: int = 5,
) -> dict:
    results = {}
    for k in range(1, max_k + 1):
        n_total = len(labels_by_seq)
        n_positive = 0
        n_negative = 0
        exact = 0
        early = 0
        ontime_or_late = 0
        miss = 0
        false_trigger_negative = 0
        delays = []
        saved_fractions_all = []
        saved_fractions_detect = []
        saved_fractions_early = []
        savings_vs_true_boundary = []

        for preds, labels, seq in zip(preds_by_seq, labels_by_seq, seqs):
            true_start = _first_index_of_class(labels, 2)
            run_start = _first_consecutive_run(preds, 2, k)
            pred_exit = None if run_start is None else run_start + k - 1
            end_positions = seq["sentence_end_positions"]
            cot_start = int(seq["cot_start_pos"])
            cot_end = int(seq["cot_end_pos"])
            total_tokens = max(cot_end - cot_start, 1)
            if true_start is None:
                n_negative += 1
                if pred_exit is not None:
                    false_trigger_negative += 1
                continue

            n_positive += 1
            true_exit_tokens_saved = cot_end - int(end_positions[true_start])
            if pred_exit is None:
                miss += 1
                saved_fractions_all.append(0.0)
                continue

            pred_exit = min(pred_exit, len(end_positions) - 1)
            saved_fraction = (cot_end - int(end_positions[pred_exit])) / total_tokens
            saved_fractions_all.append(saved_fraction)
            optimal_fraction = true_exit_tokens_saved / total_tokens
            if optimal_fraction > 0:
                savings_vs_true_boundary.append(saved_fraction / optimal_fraction)

            if pred_exit < true_start:
                early += 1
                saved_fractions_early.append(saved_fraction)
                continue

            ontime_or_late += 1
            delay = pred_exit - true_start
            delays.append(delay)
            saved_fractions_detect.append(saved_fraction)
            if delay == 0:
                exact += 1

        results[f"k={k}"] = {
            "k": k,
            "n_traces": n_total,
            "n_positive_traces": n_positive,
            "n_negative_traces": n_negative,
            "false_trigger_rate_negative": (
                false_trigger_negative / n_negative if n_negative > 0 else None
            ),
            "early_fire_rate": early / n_positive if n_positive > 0 else 0.0,
            "miss_rate": miss / n_positive if n_positive > 0 else 0.0,
            "detect_rate": ontime_or_late / n_positive if n_positive > 0 else 0.0,
            "exact_boundary_rate": exact / n_positive if n_positive > 0 else 0.0,
            "mean_delay": sum(delays) / len(delays) if delays else None,
            "median_delay": float(torch.tensor(delays, dtype=torch.float32).median().item()) if delays else None,
            "mean_saved_fraction_all": (
                sum(saved_fractions_all) / len(saved_fractions_all) if saved_fractions_all else 0.0
            ),
            "mean_saved_fraction_detect": (
                sum(saved_fractions_detect) / len(saved_fractions_detect) if saved_fractions_detect else None
            ),
            "mean_saved_fraction_early": (
                sum(saved_fractions_early) / len(saved_fractions_early) if saved_fractions_early else None
            ),
            "mean_savings_vs_true_boundary": (
                sum(savings_vs_true_boundary) / len(savings_vs_true_boundary) if savings_vs_true_boundary else None
            ),
        }
    return results


def _collect_logits(
    probe: SentenceCausalProbe,
    seqs: list,
    device: str = "cpu",
    causal_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probe.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for seq in seqs:
            logits = probe(
                seq["activations"].unsqueeze(0).to(device),
                causal_window=causal_window,
            ).squeeze(0).cpu()
            loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
            all_logits.append(logits[loss_mask])
            all_labels.append(seq["labels"].long()[loss_mask])
    if not all_logits:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_position_baseline(
    val_seqs: list,
    test_seqs: list,
    threshold_strategy: str = "f1",
) -> dict:
    def _scores_for_seq(seq: dict) -> torch.Tensor:
        loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
        n = int(loss_mask.sum().item())
        if n <= 1:
            return torch.zeros(n)
        return torch.arange(n, dtype=torch.float32) / float(n - 1)

    def _flatten(seqs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        labels = []
        for seq in seqs:
            scores.append(_scores_for_seq(seq))
            loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
            labels.append(seq["labels"].float()[loss_mask])
        return torch.cat(scores), torch.cat(labels)

    val_scores, val_labels = _flatten(val_seqs)
    test_scores, test_labels = _flatten(test_seqs)
    sweep = sweep_threshold_curve(val_scores, val_labels, low=0.0, high=1.0, step=0.05)
    strategy_key = {
        "f1": "best_f1",
        "precision_at_recall_50": "best_precision_at_recall_50",
        "precision_at_recall_30": "best_precision_at_recall_30",
    }[threshold_strategy]
    selected = sweep[strategy_key]
    threshold = selected["threshold"]
    test_preds = (test_scores > threshold).float()
    return {
        "threshold_sweep": sweep,
        "selected_threshold_strategy": threshold_strategy,
        "tuned_threshold": threshold,
        "val_selected_point": selected,
        "test_metrics": _compute_classification_metrics(test_preds, test_labels, test_scores),
    }


def compute_position_baseline_multiclass(
    val_seqs: list,
    test_seqs: list,
    step: float = 0.05,
) -> dict:
    def _scores_for_seq(seq: dict) -> torch.Tensor:
        loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
        n = int(loss_mask.sum().item())
        if n <= 1:
            return torch.zeros(n)
        return torch.arange(n, dtype=torch.float32) / float(n - 1)

    def _flatten(seqs: list) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        labels = []
        for seq in seqs:
            scores.append(_scores_for_seq(seq))
            loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
            labels.append(seq["labels"].long()[loss_mask])
        return torch.cat(scores), torch.cat(labels)

    def _predict(scores: torch.Tensor, low_th: float, high_th: float) -> torch.Tensor:
        preds = torch.ones(len(scores), dtype=torch.long)
        preds[scores <= low_th] = 0
        preds[scores > high_th] = 2
        return preds

    val_scores, val_labels = _flatten(val_seqs)
    test_scores, test_labels = _flatten(test_seqs)
    threshold_values = [round(i * step, 2) for i in range(int(1.0 / step) + 1)]
    best = None
    for low_th in threshold_values:
        for high_th in threshold_values:
            if high_th < low_th:
                continue
            preds = _predict(val_scores, low_th, high_th)
            metrics = _compute_multiclass_metrics(preds, val_labels, num_classes=3)
            candidate = {
                "low_threshold": low_th,
                "high_threshold": high_th,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
            if best is None or candidate["macro_f1"] > best["macro_f1"]:
                best = candidate
    assert best is not None
    test_preds = _predict(test_scores, best["low_threshold"], best["high_threshold"])
    return {
        "selected_threshold_strategy": "macro_f1",
        "tuned_thresholds": {
            "low_threshold": best["low_threshold"],
            "high_threshold": best["high_threshold"],
        },
        "val_selected_point": best,
        "test_metrics": _compute_multiclass_metrics(test_preds, test_labels, num_classes=3),
    }


def _compute_random_baselines_multiclass(y: torch.Tensor, seed: int = 0, num_classes: int = 3) -> dict:
    y = y.long()
    n = len(y)
    counts = torch.bincount(y, minlength=num_classes).float()
    priors = counts / counts.sum().clamp_min(1.0)
    majority_class = int(priors.argmax().item()) if n > 0 else 0
    preds_majority = torch.full((n,), majority_class, dtype=torch.long)
    rng = torch.Generator().manual_seed(seed)
    preds_rand = torch.multinomial(priors, n, replacement=True, generator=rng) if n > 0 else torch.tensor([], dtype=torch.long)
    return {
        "majority_class": _compute_multiclass_metrics(preds_majority, y, num_classes=num_classes),
        "random_prior": _compute_multiclass_metrics(preds_rand, y, num_classes=num_classes),
    }


def _print_multiclass_baselines(baselines: dict) -> None:
    print("  Baseline            Acc     MacroP  MacroR  MacroF1")
    for name, m in baselines.items():
        print(
            f"  {name:<20s}  {m['accuracy']:.4f}  {m['macro_precision']:.4f}  "
            f"{m['macro_recall']:.4f}  {m['macro_f1']:.4f}"
        )


def _collect_multiclass_predictions_by_seq(
    probe: SentenceCausalProbe,
    seqs: list,
    device: str = "cpu",
    causal_window: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    probe.eval()
    preds_by_seq = []
    labels_by_seq = []
    with torch.no_grad():
        for seq in seqs:
            logits = probe(
                seq["activations"].unsqueeze(0).to(device),
                causal_window=causal_window,
            ).squeeze(0).cpu()
            loss_mask = seq.get("loss_mask", torch.ones(len(seq["labels"]), dtype=torch.bool))
            preds_by_seq.append(logits.argmax(dim=-1)[loss_mask])
            labels_by_seq.append(seq["labels"].long().cpu()[loss_mask])
    return preds_by_seq, labels_by_seq


def train_solution_probe(
    probe: SentenceCausalProbe,
    train_seqs: list,
    val_seqs: list,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 64,
    patience: int = 6,
    device: str = "cpu",
    pos_weight: Optional[float] = None,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    early_stop_metric: str = "f1",
    task_mode: str = "three_way",
    class_weights: Optional[torch.Tensor] = None,
    causal_window: Optional[int] = None,
) -> List[dict]:
    if task_mode == "binary":
        assert early_stop_metric in ("f1", "precision", "recall", "val_loss")
    else:
        assert early_stop_metric in ("macro_f1", "accuracy", "val_loss")
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if early_stop_metric != "val_loss" else "min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    pw_tensor = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    if pos_weight is not None and task_mode == "binary":
        print(f"  Using pos_weight={pos_weight:.2f}")
    if class_weights is not None and task_mode != "binary":
        print(f"  Using class_weights={[round(float(x), 3) for x in class_weights.tolist()]}")
    print(f"  Early stopping on: {early_stop_metric}")

    higher_is_better = early_stop_metric != "val_loss"
    best_metric = float("-inf") if higher_is_better else float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(epochs):
        probe.train()
        rng = torch.Generator().manual_seed(epoch)
        perm = torch.randperm(len(train_seqs), generator=rng).tolist()
        train_loss_sum = 0.0
        train_count = 0
        for i in range(0, len(train_seqs), batch_size):
            batch = [train_seqs[j] for j in perm[i:i + batch_size]]
            H, y, m, lm = collate_sequences(batch)
            H = H.to(device)
            y = y.to(device)
            m = m.to(device)
            lm = lm.to(device)

            logits = probe(H, mask=m, causal_window=causal_window)
            if task_mode == "binary":
                loss_all = nn.BCEWithLogitsLoss(pos_weight=pw_tensor, reduction="none")(logits, y.float())
            else:
                ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None, reduction="none")
                loss_all = ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
            loss = (loss_all * lm.float()).sum() / lm.float().sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)
            optimizer.step()

            n_valid = int(lm.sum().item())
            train_loss_sum += loss.item() * n_valid
            train_count += n_valid

        train_loss = train_loss_sum / max(train_count, 1)

        probe.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for i in range(0, len(val_seqs), batch_size):
                batch = val_seqs[i:i + batch_size]
                H, y, m, lm = collate_sequences(batch)
                H = H.to(device)
                y = y.to(device)
                m = m.to(device)
                lm = lm.to(device)
                logits = probe(H, mask=m, causal_window=causal_window)
                if task_mode == "binary":
                    loss_all = nn.BCEWithLogitsLoss(pos_weight=pw_tensor, reduction="none")(logits, y.float())
                    val_preds.append((logits > 0).float()[lm].cpu())
                    val_labels.append(y[lm].cpu())
                else:
                    ce = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None, reduction="none")
                    loss_all = ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).reshape(y.shape)
                    val_preds.append(logits.argmax(dim=-1)[lm].cpu())
                    val_labels.append(y[lm].cpu())
                val_loss_sum += (loss_all * lm.float()).sum().item()
                val_count += int(lm.sum().item())

        val_loss = val_loss_sum / max(val_count, 1)
        val_preds_f = torch.cat(val_preds)
        val_labels_f = torch.cat(val_labels)
        if task_mode == "binary":
            metrics = _compute_classification_metrics(val_preds_f, val_labels_f)
        else:
            metrics = _compute_multiclass_metrics(val_preds_f, val_labels_f, num_classes=3)
        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": metrics["accuracy"],
            "lr": current_lr,
        }
        if task_mode == "binary":
            record.update({
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
            })
        else:
            record.update({
                "val_macro_precision": metrics["macro_precision"],
                "val_macro_recall": metrics["macro_recall"],
                "val_macro_f1": metrics["macro_f1"],
            })
        history.append(record)
        if task_mode == "binary":
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                  f"F1={metrics['f1']:.4f} lr={current_lr:.2e}")
        else:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"Acc={metrics['accuracy']:.4f} MacroF1={metrics['macro_f1']:.4f} lr={current_lr:.2e}")

        current_metric = {
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "macro_f1": metrics.get("macro_f1"),
            "accuracy": metrics["accuracy"],
            "val_loss": val_loss,
        }[early_stop_metric]
        scheduler.step(current_metric)
        improved = current_metric > best_metric if higher_is_better else current_metric < best_metric
        if improved:
            best_metric = current_metric
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return history


def evaluate_solution_probe(
    probe: SentenceCausalProbe,
    seqs: list,
    device: str = "cpu",
    threshold: float = 0.0,
    task_mode: str = "binary",
    causal_window: Optional[int] = None,
) -> dict:
    logits, labels = _collect_logits(probe, seqs, device=device, causal_window=causal_window)
    if len(logits) == 0:
        return {}
    if task_mode == "binary":
        preds = (logits > threshold).float()
        return _compute_classification_metrics(preds, labels.float(), logits)
    preds = logits.argmax(dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    return _compute_multiclass_metrics(preds, labels, probs=probs, num_classes=3)


def main(
    model: str,
    data_name: str,
    attr_dir: Optional[str] = None,
    attribution_glob: str = "question_*.json",
    layer: float = 0.8,
    device: str = "cuda",
    lm_device_map: Optional[str] = None,
    lr: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 64,
    patience: int = 6,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_collection: bool = False,
    skip_if_exists: bool = False,
    plain_lm: bool = False,
    clue_alpha: Optional[float] = None,
    task_mode: str = "three_way",
    sentence_aggregation: str = "last",
    max_sentences: Optional[int] = None,
    max_probe_input_tokens: Optional[int] = 256,
    max_pos_weight: float = 5.0,
    grad_clip: float = 1.0,
    early_stop_metric: str = "f1",
    threshold_strategy: str = "f1",
    hidden_dim: int = 64,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    max_traces_per_question: Optional[int] = None,
    split_file: Optional[str] = None,
    probe_train_frac: float = 0.5,
    probe_val_frac: float = 0.1,
    probe_test_frac: float = 0.1,
):
    max_probe_input_tokens = _normalize_optional_positive_int(
        max_probe_input_tokens,
        "max_probe_input_tokens",
    )
    assert sentence_aggregation in ("full", "last", "avg")
    assert task_mode in ("binary", "three_way")
    if task_mode == "binary":
        assert threshold_strategy in ("f1", "precision_at_recall_50", "precision_at_recall_30")
    else:
        if threshold_strategy != "f1":
            print(f"Ignoring threshold_strategy={threshold_strategy} for task_mode={task_mode}; multiclass uses argmax.")
        if early_stop_metric == "f1":
            print("Using early_stop_metric=macro_f1 for task_mode=three_way.")
            early_stop_metric = "macro_f1"
    if clue_alpha is not None:
        print(
            "WARNING: train_solution_probe --clue_alpha is deprecated and ignored. "
            "Set --clue_alpha when running sentence-causal attribution instead."
        )

    model_short = model.split("/")[-1]
    data_short = data_name.split("/")[-1]
    if attr_dir is None:
        attr_dir = str(
            Path("outputs") / model_short / data_short / "contribution_graphs" / "sentence_causal" / "boxed"
        )
    print(f"Attribution dir: {attr_dir}")
    trace_labels = load_solution_attribution_data(attr_dir, attribution_glob)
    traces_data = get_reasoning_traces(model, data_name)

    from attributions.utils import get_thinking_tokens
    end_ids = None
    if not plain_lm:
        thinking = get_thinking_tokens(model)
        end_ids = thinking.get("end_token_ids")
        if end_ids is None:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            end_ids = _tok.encode(thinking["end_token"], add_special_tokens=False)
            del _tok
        print(f"End-thinking token IDs: {end_ids}")

    _cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir(model_short, data_short)
    if not skip_collection:
        can_check_layer_without_model = not (0 < float(layer) < 1) and int(layer) >= 0
        if can_check_layer_without_model:
            resolved_layer = int(layer)
            _complete, _msg = _check_cache_complete_for_layer(_cache_dir, trace_labels, resolved_layer)
            if _complete:
                print(f"Cache complete ({_msg}) — skipping activation collection.")
                skip_collection = True
                layer = resolved_layer
            elif _cache_dir.exists() and any(_cache_dir.glob("*.pt")):
                print(f"Partial cache: {_msg}.")
        elif _cache_dir.exists() and any(_cache_dir.glob("*.pt")):
            print(
                "Layer requires model resolution before cache completeness can be checked; "
                "will verify missing files after loading the model."
            )

    if not skip_collection:
        print(f"Loading model: {model}")
        lm = load_nnsight_model(model, device=device, device_map=lm_device_map)
        cfg = detect_model_config(lm)
        print(f"Model: {cfg.n_layers} layers, d_model={cfg.d_model}")
        if 0 < layer < 1:
            layer = int(round(layer * cfg.n_layers))
        else:
            layer = int(layer)
        if layer < 0:
            layer = int(cfg.n_layers + layer)
        print(f"Probing layer {layer}")
        collect_and_cache(
            lm, cfg, trace_labels, traces_data,
            layer=layer, cache_dir=_cache_dir, device=device,
            plain_lm=plain_lm, end_ids=end_ids,
        )
        d_model = cfg.d_model
        del lm, cfg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if 0 < layer < 1:
            raise ValueError("Fractional layer requires model loading when using --skip_collection")
        layer = int(layer)
        if layer == -1:
            cache_files = list(_cache_dir.glob("*.pt"))
            if not cache_files:
                raise FileNotFoundError(f"No cache files in {_cache_dir}")
            layers_found = set()
            for f in cache_files:
                parts = f.stem.split("_L")
                if len(parts) == 2:
                    layers_found.add(int(parts[1]))
            layer = max(layers_found)
        sample = torch.load(next(_cache_dir.glob(f"*_L{layer}.pt")), weights_only=True)
        d_model = sample["activations"].shape[1]

    print(f"d_model = {d_model}, layer = {layer}")
    print(f"Context policy: {CONTEXT_POLICY}")
    print(f"Causal lookback window: {max_probe_input_tokens if max_probe_input_tokens is not None else 'full prefix'}")

    default_output_name = (
        f"solution_probe_{sentence_aggregation}"
        if task_mode == "binary"
        else f"solution_probe_{task_mode}_{sentence_aggregation}"
    )
    effective_output_dir = Path(output_dir) if output_dir else Path(attr_dir) / default_output_name
    if skip_if_exists:
        results_path = effective_output_dir / f"solution_probe_results_L{layer}.json"
        if results_path.exists():
            with open(results_path) as f:
                existing_results = json.load(f)
            existing_hparams = existing_results.get("hyperparams", {})
            if (
                existing_results.get("context_policy") == CONTEXT_POLICY
                and existing_hparams.get("max_probe_input_tokens") == max_probe_input_tokens
                and existing_hparams.get("sentence_aggregation") == sentence_aggregation
                and existing_hparams.get("task_mode") == task_mode
            ):
                print(f"Compatible results already exist at {results_path} — skipping.")
                return
            print(
                f"Existing results at {results_path} are incompatible with "
                f"context_policy={CONTEXT_POLICY}; retraining."
            )

    split_path = Path(split_file) if split_file else Path(attr_dir) / "question_split.json"
    max_label_qid = max((qid for qid, _t_idx in trace_labels.keys()), default=-1)
    n_total = max(len(traces_data), max_label_qid + 1)
    if len(traces_data) < n_total:
        print(
            f"WARNING: teacher traces file has {len(traces_data)} questions, but attribution labels "
            f"cover {n_total}; using attribution question IDs for probe split."
        )
    split_dict = generate_question_split(
        n_total=n_total,
        n_train=max(1, int(n_total * probe_train_frac)),
        n_val=max(1, int(n_total * probe_val_frac)),
        n_test=max(1, int(n_total * probe_test_frac)),
        seed=seed,
        save_path=split_path,
    )
    print(f"\nQuestion-level split (seed={seed}, n_total={n_total}):")
    print(f"  Train: {len(split_dict['probe_train'])} questions")
    print(f"  Val:   {len(split_dict['probe_val'])} questions")
    print(f"  Test:  {len(split_dict['probe_test'])} questions")
    print(f"  Eval:  {len(split_dict['eval'])} questions")
    print(f"  Split loaded from: {split_path}")

    print("\nPreparing sentence-state datasets...")
    train_seqs, val_seqs, test_seqs, stats = prepare_solution_datasets(
        _cache_dir,
        trace_labels,
        int(layer),
        split_dict["probe_train"],
        split_dict["probe_val"],
        split_dict["probe_test"],
        seed=seed,
        sentence_aggregation=sentence_aggregation,
        task_mode=task_mode,
        max_sentences=max_sentences,
        max_probe_input_tokens=max_probe_input_tokens,
        max_traces_per_question=max_traces_per_question,
    )

    n_classes = 1 if task_mode == "binary" else 3
    probe = SentenceCausalProbe(d_model, hidden_dim=hidden_dim, dropout=dropout, n_classes=n_classes)
    print(f"\nProbe: SentenceCausalProbe ({sum(p.numel() for p in probe.parameters())} params)")
    train_device = device
    if train_device != "cpu" and not torch.cuda.is_available():
        print(f"CUDA unavailable, falling back to CPU for training (requested {device}).")
        train_device = "cpu"

    pos_weight = stats.get("pos_weight")
    if pos_weight is not None and pos_weight > max_pos_weight:
        print(f"  Capping pos_weight {pos_weight:.2f} -> {max_pos_weight:.2f}")
        pos_weight = max_pos_weight
    class_weights = _compute_multiclass_class_weights(train_seqs, num_classes=3) if task_mode == "three_way" else None

    print(f"\nTraining on {train_device}...")
    history = train_solution_probe(
        probe, train_seqs, val_seqs,
        lr=lr, epochs=epochs, batch_size=batch_size, patience=patience,
        device=train_device, pos_weight=pos_weight, weight_decay=weight_decay,
        grad_clip=grad_clip, early_stop_metric=early_stop_metric,
        task_mode=task_mode, class_weights=class_weights,
        causal_window=max_probe_input_tokens,
    )

    if task_mode == "binary":
        print("\nThreshold sweep on validation set:")
        val_logits, val_labels = _collect_logits(
            probe,
            val_seqs,
            device=train_device,
            causal_window=max_probe_input_tokens,
        )
        sweep = sweep_threshold_curve(val_logits, val_labels.float(), low=-3.0, high=5.0, step=0.1)
        print(f"  {'Thresh':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
        for pt in sweep["pr_curve"]:
            marker = ""
            if pt["threshold"] == sweep["best_f1"]["threshold"]:
                marker += " <-- best F1"
            if pt["threshold"] == sweep["best_precision_at_recall_50"]["threshold"]:
                marker += " <-- best P@R≥0.5"
            if pt["threshold"] == sweep["best_precision_at_recall_30"]["threshold"]:
                marker += " <-- best P@R≥0.3"
            print(f"  {pt['threshold']:6.1f}  {pt['precision']:6.4f}  "
                  f"{pt['recall']:6.4f}  {pt['f1']:6.4f}{marker}")

        strategy_key = {
            "f1": "best_f1",
            "precision_at_recall_50": "best_precision_at_recall_50",
            "precision_at_recall_30": "best_precision_at_recall_30",
        }[threshold_strategy]
        selected = sweep[strategy_key]
        tuned_threshold = selected["threshold"]
        print(f"\n  Selected threshold ({threshold_strategy}): {tuned_threshold:.1f} "
              f"(P={selected['precision']:.4f} R={selected['recall']:.4f} F1={selected['f1']:.4f})")

        print("\nSentence-position baseline:")
        position_baseline = compute_position_baseline(val_seqs, test_seqs, threshold_strategy=threshold_strategy)
        pb_sel = position_baseline["val_selected_point"]
        pb_test = position_baseline["test_metrics"]
        print(f"  Selected threshold ({threshold_strategy}): {position_baseline['tuned_threshold']:.2f} "
              f"(val P={pb_sel['precision']:.4f} R={pb_sel['recall']:.4f} F1={pb_sel['f1']:.4f})")
        print(f"  Test: P={pb_test['precision']:.4f} R={pb_test['recall']:.4f} "
              f"F1={pb_test['f1']:.4f} AUROC={pb_test.get('auroc', 'N/A')}")

        print("\nRandom baselines (test set):")
        test_labels_flat = torch.cat([
            s["labels"].float()[s.get("loss_mask", torch.ones(len(s["labels"]), dtype=torch.bool))]
            for s in test_seqs
        ])
        baselines = _compute_random_baselines(test_labels_flat, seed=seed)
        _print_baselines(baselines)

        print(f"\nTest set evaluation (threshold={tuned_threshold:.1f}):")
        metrics = evaluate_solution_probe(
            probe,
            test_seqs,
            device=train_device,
            threshold=tuned_threshold,
            task_mode=task_mode,
            causal_window=max_probe_input_tokens,
        )
        n_pos = metrics.get("tp", 0) + metrics.get("fn", 0)
        n_neg = metrics.get("fp", 0) + metrics.get("tn", 0)
        print(f"  Samples:   {n_pos + n_neg} ({n_pos} pos, {n_neg} neg)")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1:        {metrics.get('f1', 0):.4f}")
        if metrics.get("auroc") is not None:
            print(f"  AUROC:     {metrics['auroc']:.4f}")
    else:
        sweep = None
        tuned_threshold = None
        print("\nSentence-position baseline:")
        position_baseline = compute_position_baseline_multiclass(val_seqs, test_seqs)
        pb_sel = position_baseline["val_selected_point"]
        pb_test = position_baseline["test_metrics"]
        tuned_pair = position_baseline["tuned_thresholds"]
        print(
            f"  Selected thresholds (macro_f1): low={tuned_pair['low_threshold']:.2f} "
            f"high={tuned_pair['high_threshold']:.2f} (val Acc={pb_sel['accuracy']:.4f} MacroF1={pb_sel['macro_f1']:.4f})"
        )
        print(
            f"  Test: Acc={pb_test['accuracy']:.4f} MacroF1={pb_test['macro_f1']:.4f} "
            f"MacroP={pb_test['macro_precision']:.4f} MacroR={pb_test['macro_recall']:.4f}"
        )

        print("\nRandom baselines (test set):")
        test_labels_flat = torch.cat([
            s["labels"].long()[s.get("loss_mask", torch.ones(len(s["labels"]), dtype=torch.bool))]
            for s in test_seqs
        ])
        baselines = _compute_random_baselines_multiclass(test_labels_flat, seed=seed)
        _print_multiclass_baselines(baselines)

        print("\nTest set evaluation (argmax):")
        metrics = evaluate_solution_probe(
            probe,
            test_seqs,
            device=train_device,
            task_mode=task_mode,
            causal_window=max_probe_input_tokens,
        )
        preds_by_seq, labels_by_seq = _collect_multiclass_predictions_by_seq(
            probe,
            test_seqs,
            device=train_device,
            causal_window=max_probe_input_tokens,
        )
        boundary_metrics = compute_class2_boundary_metrics(preds_by_seq, labels_by_seq, test_seqs, max_k=5)
        print(f"  Samples:   {metrics.get('support', 0)}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  MacroP:    {metrics.get('macro_precision', 0):.4f}")
        print(f"  MacroR:    {metrics.get('macro_recall', 0):.4f}")
        print(f"  MacroF1:   {metrics.get('macro_f1', 0):.4f}")
        if metrics.get("auroc_macro_ovr") is not None:
            print(f"  AUROC:     {metrics['auroc_macro_ovr']:.4f}")
        print("  Confusion matrix [true][pred]:")
        for row in metrics.get("confusion_matrix", []):
            print(f"    {row}")
        print("  Per-class:")
        for cls in metrics.get("per_class", []):
            print(
                f"    class {cls['class_id']}: P={cls['precision']:.4f} "
                f"R={cls['recall']:.4f} F1={cls['f1']:.4f} support={cls['support']}"
            )
        print("  Class-2 boundary diagnostics:")
        for key, bm in boundary_metrics.items():
            mean_delay = "N/A" if bm["mean_delay"] is None else f"{bm['mean_delay']:.2f}"
            median_delay = "N/A" if bm["median_delay"] is None else f"{bm['median_delay']:.2f}"
            neg_fp = "N/A" if bm["false_trigger_rate_negative"] is None else f"{bm['false_trigger_rate_negative']:.4f}"
            saved_all = f"{100.0 * bm['mean_saved_fraction_all']:.2f}%"
            saved_detect = "N/A" if bm["mean_saved_fraction_detect"] is None else f"{100.0 * bm['mean_saved_fraction_detect']:.2f}%"
            saved_early = "N/A" if bm["mean_saved_fraction_early"] is None else f"{100.0 * bm['mean_saved_fraction_early']:.2f}%"
            saved_vs_true = "N/A" if bm["mean_savings_vs_true_boundary"] is None else f"{100.0 * bm['mean_savings_vs_true_boundary']:.2f}%"
            print(
                f"    {key}: detect={bm['detect_rate']:.4f} early={bm['early_fire_rate']:.4f} "
                f"miss={bm['miss_rate']:.4f} exact={bm['exact_boundary_rate']:.4f} "
                f"mean_delay={mean_delay} median_delay={median_delay} neg_fp={neg_fp} "
                f"saved_all={saved_all} saved_detect={saved_detect} saved_early={saved_early} "
                f"vs_true={saved_vs_true}"
            )

    effective_output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": model,
        "data_name": data_name,
        "layer": layer,
        "probe_type": "solution_sentence_causal",
        "context_policy": CONTEXT_POLICY,
        "task": "has_plausible_solution_in_mind" if task_mode == "binary" else "three_way_solution_state",
        "dataset_stats": stats,
        "hyperparams": {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
            "task_mode": task_mode,
            "sentence_aggregation": sentence_aggregation,
            "context_policy": CONTEXT_POLICY,
            "max_sentences": max_sentences,
            "max_probe_input_tokens": max_probe_input_tokens,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "max_pos_weight": max_pos_weight,
            "grad_clip": grad_clip,
            "early_stop_metric": early_stop_metric,
            "threshold_strategy": threshold_strategy,
            "probe_train_frac": probe_train_frac,
            "probe_val_frac": probe_val_frac,
            "probe_test_frac": probe_test_frac,
            "max_traces_per_question": max_traces_per_question,
        },
        "threshold_sweep": sweep,
        "tuned_threshold": tuned_threshold,
        "history": history,
        "test_metrics": metrics,
        "position_baseline": position_baseline,
        "random_baselines": baselines,
    }
    if task_mode == "three_way":
        results["class2_boundary_metrics"] = boundary_metrics

    with open(effective_output_dir / f"solution_probe_results_L{layer}.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(probe.state_dict(), effective_output_dir / f"solution_probe_weights_L{layer}.pt")
    print(f"\nResults saved to {effective_output_dir / f'solution_probe_results_L{layer}.json'}")
    print(f"Weights saved to {effective_output_dir / f'solution_probe_weights_L{layer}.pt'}")


if __name__ == "__main__":
    fire.Fire(main)
