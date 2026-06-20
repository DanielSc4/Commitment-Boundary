#!/usr/bin/env python3
"""Precollect sentence-causal solution-probe activations for multiple layers."""

import gc
from pathlib import Path
from typing import List, Optional

import fire
import torch

from attributions.train_solution_probe import (
    collect_and_cache,
    load_solution_attribution_data,
)
from attributions.modeling import detect_model_config, load_nnsight_model
from attributions.utils import get_reasoning_traces, get_thinking_tokens


def _parse_layers(value) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _resolve_layer(layer: float, n_layers: int) -> int:
    if 0 < layer < 1:
        resolved = int(round(layer * n_layers))
    else:
        resolved = int(layer)
    if resolved < 0:
        resolved = int(n_layers + resolved)
    if resolved < 0 or resolved >= n_layers:
        raise ValueError(f"Layer {layer} resolves to invalid layer {resolved} for {n_layers} layers")
    return resolved


def main(
    model: str,
    data_name: str,
    layers: str,
    attr_dir: Optional[str] = None,
    attribution_glob: str = "question_*.json",
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    plain_lm: bool = False,
):
    model_short = model.split("/")[-1]
    data_short = data_name.split("/")[-1]
    if attr_dir is None:
        attr_dir = str(
            Path("outputs")
            / model_short
            / data_short
            / "contribution_graphs"
            / "sentence_causal"
            / "boxed"
        )
    if cache_dir is None:
        cache_dir = str(Path("outputs") / model_short / data_short / "probe_cache")

    print(f"Attribution dir: {attr_dir}")
    print(f"Cache dir:       {cache_dir}")
    trace_labels = load_solution_attribution_data(attr_dir, attribution_glob)
    traces_data = get_reasoning_traces(model, data_name)

    end_ids = None
    if not plain_lm:
        thinking = get_thinking_tokens(model)
        end_ids = thinking.get("end_token_ids")
        if end_ids is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            end_ids = tok.encode(thinking["end_token"], add_special_tokens=False)
            del tok
        print(f"End-thinking token IDs: {end_ids}")

    print(f"Loading model: {model}")
    lm = load_nnsight_model(model, device=device)
    cfg = detect_model_config(lm)
    resolved_layers = sorted({_resolve_layer(layer, cfg.n_layers) for layer in _parse_layers(layers)})
    print(f"Model: {cfg.n_layers} layers, d_model={cfg.d_model}")
    print(f"Collecting layers: {resolved_layers}")

    collect_and_cache(
        lm,
        cfg,
        trace_labels,
        traces_data,
        layer=resolved_layers,
        cache_dir=Path(cache_dir),
        device=device,
        plain_lm=plain_lm,
        end_ids=end_ids,
    )

    del lm, cfg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fire.Fire(main)
