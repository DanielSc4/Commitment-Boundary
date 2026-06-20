"""Activation collection for sentence-level commitment probes."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from attributions.modeling import ModelConfig, find_subsequence


def get_cot_unit_metadata(
    trace_info: dict,
    prompt_len: int,
    plain_lm: bool = False,
    end_ids: Optional[List[int]] = None,
) -> dict:
    """Resolve the CoT token region and sentence endpoints for one trace."""
    target_pos = trace_info["target_pos"]
    raw_scores = trace_info["unit_scores"]
    cot_start = 0 if plain_lm else prompt_len

    if end_ids is not None and not plain_lm:
        end_pos = find_subsequence(trace_info["full_ids"], end_ids)
        cot_end = end_pos + len(end_ids) if end_pos >= 0 else target_pos
    else:
        cot_end = target_pos

    cot_positions = list(range(cot_start, cot_end))
    granularity = trace_info.get("granularity", "token")
    if granularity != "sentence":
        raise ValueError("The publication pipeline only supports sentence-level attributions.")

    unit_spans = []
    unit_scores = []
    unit_probs = []
    for index, span in enumerate(trace_info.get("unit_spans") or []):
        start = max(int(span["start_pos"]), cot_start)
        end = min(int(span["end_pos"]), cot_end)
        if start >= end:
            continue
        end_token_pos = min(int(span.get("end_token_pos", end - 1)), end - 1)
        unit_spans.append(
            {
                "start_pos": start,
                "end_pos": end,
                "end_token_pos": end_token_pos,
                "text": span.get("text", ""),
            }
        )
        unit_scores.append(float(raw_scores[index]))
        if trace_info.get("unit_probs") is not None:
            unit_probs.append(float(trace_info["unit_probs"][index]))

    unit_end_positions = [span["end_token_pos"] for span in unit_spans]
    return {
        "granularity": granularity,
        "cot_start": cot_start,
        "cot_end": cot_end,
        "cot_positions": cot_positions,
        "unit_scores": unit_scores,
        "unit_probs": unit_probs if unit_probs else None,
        "unit_spans": unit_spans,
        "unit_spans_local": [
            [span["start_pos"] - cot_start, span["end_pos"] - cot_start]
            for span in unit_spans
        ],
        "unit_end_positions": unit_end_positions,
        "unit_end_indices": [position - cot_start for position in unit_end_positions],
    }


def collect_and_cache(
    lm,
    cfg: ModelConfig,
    labels: Dict[Tuple[int, int], dict],
    traces_data: list,
    layer,
    cache_dir: Path,
    device: str,
    plain_lm: bool = False,
    end_ids: Optional[List[int]] = None,
) -> None:
    """Collect one residual-stream activation per CoT token for selected layers."""
    layers = [layer] if isinstance(layer, int) else list(layer)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def prompt_len_for_trace(question_id: int, trace_info: dict) -> Optional[int]:
        if question_id < len(traces_data):
            return len(traces_data[question_id]["prompt_tokens"])
        spans = trace_info.get("unit_spans") or []
        return min((int(span["start_pos"]) for span in spans), default=None)

    todo = {
        key: [
            selected_layer
            for selected_layer in layers
            if not (cache_dir / f"q{key[0]}_t{key[1]}_L{selected_layer}.pt").exists()
        ]
        for key in labels
    }
    todo = {key: missing for key, missing in todo.items() if missing}
    if not todo:
        print(f"Activation cache is complete in {cache_dir}")
        return

    for (question_id, trace_index), missing_layers in tqdm(
        todo.items(), desc="Collecting activations", unit="trace", dynamic_ncols=True
    ):
        trace_info = labels[(question_id, trace_index)]
        prompt_len = prompt_len_for_trace(question_id, trace_info)
        if prompt_len is None:
            tqdm.write(f"WARNING: q{question_id}_t{trace_index}: prompt length unavailable")
            continue
        metadata = get_cot_unit_metadata(
            trace_info, prompt_len, plain_lm=plain_lm, end_ids=end_ids
        )
        if not metadata["unit_end_indices"]:
            continue

        full_ids = torch.tensor([trace_info["full_ids"]], device=device)
        cot_positions = torch.tensor(metadata["cot_positions"])
        try:
            saves = {}
            with torch.no_grad():
                with lm.trace(full_ids):
                    for selected_layer in missing_layers:
                        saves[selected_layer] = (
                            cfg.get_block(lm, selected_layer).output[0].save()
                        )

            for selected_layer, saved in saves.items():
                activations = saved.value if hasattr(saved, "value") else saved
                if activations.dim() == 2:
                    activations = activations.unsqueeze(0)
                activations = activations.cpu()
                cot_activations = activations[0, cot_positions, :]
                cache_entry = {
                    "activations": cot_activations.half(),
                    "positions": metadata["cot_positions"],
                    "granularity": "sentence",
                    "unit_scores": torch.tensor(metadata["unit_scores"], dtype=torch.float32),
                    "unit_end_indices": metadata["unit_end_indices"],
                    "unit_end_positions": metadata["unit_end_positions"],
                    "unit_spans": metadata["unit_spans_local"],
                }
                if metadata["unit_probs"] is not None:
                    cache_entry["unit_probs"] = torch.tensor(
                        metadata["unit_probs"], dtype=torch.float32
                    )
                torch.save(
                    cache_entry,
                    cache_dir / f"q{question_id}_t{trace_index}_L{selected_layer}.pt",
                )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            tqdm.write(f"q{question_id}_t{trace_index}: out of memory; skipped")
        finally:
            del full_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
