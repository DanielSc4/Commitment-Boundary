"""Threshold logic and probe filename builders.

Central place for converting raw GIM scores to binary labels and for
constructing threshold-aware filenames used by training and experiment scripts.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# GIM score thresholding
# ---------------------------------------------------------------------------


def apply_threshold(
    scores: torch.Tensor,
    threshold_constant: Optional[float] = None,
    threshold_topp: Optional[float] = None,
    granularity: str = "token",
) -> torch.Tensor:
    """Convert raw scores to binary labels using a threshold strategy.

    Args:
        scores: Raw scores (float tensor).
            - token granularity: GIM scores (unbounded gradient-based values).
            - sentence granularity: importance_delta values (probability differences,
              roughly in [-1, 1]).
        threshold_constant: Units where score >= constant get label 1.
        threshold_topp: Threshold semantics depend on granularity:
            - token: top x% of scores by rank get label 1 (e.g. 10 = top 10%).
            - sentence: units where importance_delta >= threshold_topp/100 get
              label 1 (e.g. 20 → delta >= 0.20, i.e. ≥20pp probability increase).
        granularity: "token" or "sentence". Controls threshold_topp semantics.

    Returns:
        Binary labels (long tensor, same length as scores).
    """
    if threshold_constant is not None:
        return (scores >= threshold_constant).long()
    elif threshold_topp is not None:
        if granularity == "sentence":
            # Absolute delta threshold: topp=20 means importance_delta >= 0.20
            return (scores >= threshold_topp / 100.0).long()
        else:
            # Rank-based: top x% of scores
            k = max(1, int(len(scores) * threshold_topp / 100.0))
            topk_val = scores.topk(k).values[-1]
            return (scores >= topk_val).long()
    else:
        raise ValueError("One of threshold_constant or threshold_topp must be specified")


# ---------------------------------------------------------------------------
# Threshold suffix for filenames
# ---------------------------------------------------------------------------


def threshold_suffix(
    threshold_constant: Optional[float] = None,
    threshold_topp: Optional[float] = None,
) -> str:
    """Build a filesystem-safe suffix encoding the threshold parameters.

    Examples:
        threshold_topp=5    -> "_topp05"
        threshold_topp=10   -> "_topp10"
        threshold_topp=2.5  -> "_topp2p5"
        threshold_constant=0.06 -> "_const006"
        threshold_constant=0.1  -> "_const010"
    """
    if threshold_topp is not None:
        if threshold_topp == int(threshold_topp):
            return f"_topp{int(threshold_topp):02d}"
        # Fractional: replace '.' with 'p'
        return f"_topp{str(threshold_topp).replace('.', 'p')}"
    elif threshold_constant is not None:
        # Common case: multiply by 100 if result is integer, zero-pad to 3
        # e.g. 0.06 -> 006, 0.1 -> 010, 0.005 -> 0p5
        val_x100 = threshold_constant * 100
        if val_x100 == int(val_x100):
            return f"_const{int(val_x100):03d}"
        # Unusual value (e.g. 0.005): replace '.' with 'p'
        return f"_const{str(threshold_constant).replace('.', 'p')}"
    else:
        raise ValueError("One of threshold_constant or threshold_topp must be specified")


# ---------------------------------------------------------------------------
# Probe filename builders
# ---------------------------------------------------------------------------


def probe_weights_filename(
    layer: int,
    lookahead_k: int = 0,
    threshold_constant: Optional[float] = None,
    threshold_topp: Optional[float] = None,
    prefix: str = "probe",
) -> str:
    """Build probe weights filename with threshold suffix.

    Examples:
        probe_weights_L19_k0_topp05.pt
        attn_probe_weights_L19_const006.pt
    """
    sfx = threshold_suffix(threshold_constant, threshold_topp)
    if prefix == "attn_probe":
        return f"attn_probe_weights_L{layer}{sfx}.pt"
    return f"probe_weights_L{layer}_k{lookahead_k}{sfx}.pt"


def probe_results_filename(
    layer: int,
    lookahead_k: int = 0,
    threshold_constant: Optional[float] = None,
    threshold_topp: Optional[float] = None,
    prefix: str = "probe",
) -> str:
    """Build probe results filename with threshold suffix.

    Examples:
        probe_results_L19_k0_topp05.json
        attn_probe_results_L19_const006.json
    """
    sfx = threshold_suffix(threshold_constant, threshold_topp)
    if prefix == "attn_probe":
        return f"attn_probe_results_L{layer}{sfx}.json"
    return f"probe_results_L{layer}_k{lookahead_k}{sfx}.json"
