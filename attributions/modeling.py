"""Focused NNsight helpers used by probe training and early-exit evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from nnsight import LanguageModel

try:
    from nnsight import VisionLanguageModel
except ImportError:  # Older NNsight releases expose only LanguageModel.
    VisionLanguageModel = LanguageModel


@dataclass
class ModelConfig:
    """Accessors for the decoder blocks of supported Hugging Face model families."""

    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    wo_is_conv1d: bool
    mlp_output_is_tuple: bool = False
    get_block: Callable = field(repr=False, default=None)
    get_attn: Callable = field(repr=False, default=None)
    get_mlp: Callable = field(repr=False, default=None)
    get_fused_qkv: Optional[Callable] = field(repr=False, default=None)
    get_v_proj: Optional[Callable] = field(repr=False, default=None)
    get_W_O: Callable = field(repr=False, default=None)


def load_nnsight_model(
    model: str,
    device: str = "cuda",
    device_map: Optional[str] = None,
):
    """Load a text or multimodal checkpoint through the appropriate NNsight wrapper."""
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    is_multimodal = (
        getattr(hf_config, "model_type", None)
        in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    )
    model_cls = VisionLanguageModel if is_multimodal else LanguageModel
    lm = model_cls(
        model,
        device_map=device_map or device,
        dispatch=True,
        attn_implementation="eager",
        trust_remote_code=True,
        dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    )
    lm.eval()
    return lm


def detect_model_config(lm) -> ModelConfig:
    """Map a supported model family to its decoder internals."""
    hf_config = lm.config
    model_type = getattr(hf_config, "model_type", None)
    if model_type == "gpt2":
        return _gpt2_config(lm, hf_config)
    if model_type in ("llama", "mistral", "qwen3"):
        return _llama_config(lm, hf_config)
    if model_type == "gpt_oss":
        return _gpt_oss_config(lm, hf_config)
    if model_type == "gemma4":
        return _gemma4_config(lm, hf_config)
    raise ValueError(
        f"Unsupported model_type={model_type!r}. "
        "Supported: gpt2, llama, mistral, qwen3, gpt_oss, gemma4."
    )


def _gpt2_config(lm, cfg: Any) -> ModelConfig:
    n_heads = cfg.n_head
    return ModelConfig(
        n_layers=cfg.n_layer,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        d_model=cfg.n_embd,
        d_head=cfg.n_embd // n_heads,
        wo_is_conv1d=True,
        get_block=lambda model, layer: model.transformer.h[layer],
        get_attn=lambda model, layer: model.transformer.h[layer].attn,
        get_mlp=lambda model, layer: model.transformer.h[layer].mlp,
        get_fused_qkv=lambda model, layer: model.transformer.h[layer].attn.c_attn,
        get_W_O=lambda layer: lm.transformer.h[layer].attn.c_proj.weight.data,
    )


def _llama_config(lm, cfg: Any) -> ModelConfig:
    n_heads = cfg.num_attention_heads
    return ModelConfig(
        n_layers=cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=getattr(cfg, "num_key_value_heads", n_heads),
        d_model=cfg.hidden_size,
        d_head=getattr(cfg, "head_dim", cfg.hidden_size // n_heads),
        wo_is_conv1d=False,
        get_block=lambda model, layer: model.model.layers[layer],
        get_attn=lambda model, layer: model.model.layers[layer].self_attn,
        get_mlp=lambda model, layer: model.model.layers[layer].mlp,
        get_v_proj=lambda model, layer: model.model.layers[layer].self_attn.v_proj,
        get_W_O=lambda layer: lm.model.layers[layer].self_attn.o_proj.weight.data,
    )


def _gemma4_config(lm, cfg: Any) -> ModelConfig:
    text_cfg = cfg.text_config
    n_heads = text_cfg.num_attention_heads
    return ModelConfig(
        n_layers=text_cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=getattr(text_cfg, "num_key_value_heads", n_heads),
        d_model=text_cfg.hidden_size,
        d_head=getattr(text_cfg, "head_dim", text_cfg.hidden_size // n_heads),
        wo_is_conv1d=False,
        get_block=lambda model, layer: model.model.language_model.layers[layer],
        get_attn=lambda model, layer: model.model.language_model.layers[layer].self_attn,
        get_mlp=lambda model, layer: model.model.language_model.layers[layer].mlp,
        get_v_proj=lambda model, layer: model.model.language_model.layers[layer].self_attn.v_proj,
        get_W_O=lambda layer: lm.model.language_model.layers[layer].self_attn.o_proj.weight.data,
    )


def _gpt_oss_config(lm, cfg: Any) -> ModelConfig:
    n_heads = cfg.num_attention_heads
    return ModelConfig(
        n_layers=cfg.num_hidden_layers,
        n_heads=n_heads,
        n_kv_heads=getattr(cfg, "num_key_value_heads", n_heads),
        d_model=cfg.hidden_size,
        d_head=getattr(cfg, "head_dim", cfg.hidden_size // n_heads),
        wo_is_conv1d=False,
        mlp_output_is_tuple=True,
        get_block=lambda model, layer: model.model.layers[layer],
        get_attn=lambda model, layer: model.model.layers[layer].self_attn,
        get_mlp=lambda model, layer: model.model.layers[layer].mlp,
        get_v_proj=lambda model, layer: model.model.layers[layer].self_attn.v_proj,
        get_W_O=lambda layer: lm.model.layers[layer].self_attn.o_proj.weight.data,
    )


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the first index of ``needle`` in ``haystack``, or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index
    return -1


@torch.no_grad()
def greedy_generate(
    lm,
    tokens: torch.Tensor,
    max_new_tokens: int = 10,
    eos_token_id: Optional[int] = None,
) -> list[int]:
    """Greedily decode from an already-tokenized prefix."""
    generated: list[int] = []
    current = tokens
    for _ in range(max_new_tokens):
        with lm.trace(current):
            logits = lm.output.logits.save()
        next_id = int(logits[0, -1].argmax().item())
        if next_id == eos_token_id:
            break
        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)
        current = torch.cat([current, next_token], dim=1)
    return generated


def collect_residual_stream(lm, cfg: ModelConfig, tokens: torch.Tensor) -> torch.Tensor:
    """Collect decoder block outputs with shape ``[layers, sequence, hidden]``."""
    saves = {}
    with torch.no_grad():
        with lm.trace(tokens):
            for layer in range(cfg.n_layers):
                saves[layer] = cfg.get_block(lm, layer).output[0].save()

    layers = []
    for layer in range(cfg.n_layers):
        value = saves[layer].value if hasattr(saves[layer], "value") else saves[layer]
        if value.dim() == 2:
            value = value.unsqueeze(0)
        layers.append(value[0])
    return torch.stack(layers, dim=0)
