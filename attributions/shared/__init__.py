"""Shared utilities for probe training and compression experiments."""

from .threshold import (
    apply_threshold,
    threshold_suffix,
    probe_weights_filename,
    probe_results_filename,
)
from .metrics import (
    compute_classification_metrics,
    compute_random_baselines,
    print_baselines,
    sweep_threshold_curve,
)
from .cache import (
    parse_cache_question_id,
    check_cache_complete,
    bucket_files_by_question,
)
from .sentence import (
    SentenceSpan,
    split_sentences_from_token_ids,
    expand_sentence_spans,
)

__all__ = [
    "apply_threshold",
    "threshold_suffix",
    "probe_weights_filename",
    "probe_results_filename",
    "compute_classification_metrics",
    "compute_random_baselines",
    "print_baselines",
    "sweep_threshold_curve",
    "parse_cache_question_id",
    "check_cache_complete",
    "bucket_files_by_question",
    "SentenceSpan",
    "split_sentences_from_token_ids",
    "expand_sentence_spans",
]
