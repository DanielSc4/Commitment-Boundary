"""Cache file parsing, completeness checks, and question-level bucketing.

Shared by train_probe.py and train_attn_probe.py.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_cache_question_id(filename: str) -> Optional[int]:
    """Extract question ID from a cache filename like q{id}_t{idx}_L{layer}.pt."""
    m = re.match(r"q(\d+)_t\d+_L\d+\.pt", Path(filename).name)
    return int(m.group(1)) if m else None


def check_cache_complete(cache_dir: Path, labels: dict) -> Tuple[bool, str]:
    """Check whether every (q_id, t_idx) in labels has a cache file.

    Layer is intentionally ignored: for fractional layers the integer index
    can't be resolved without the model, so we check only question/trace identity.
    If collection was interrupted, the missing traces are reported.

    Returns:
        (is_complete, message) -- message is suitable for printing.
    """
    if not cache_dir.exists():
        return False, "cache directory does not exist"

    cached_pairs: set = set()
    for f in cache_dir.glob("*.pt"):
        m = re.match(r"q(\d+)_t(\d+)_L\d+\.pt", f.name)
        if m:
            cached_pairs.add((int(m.group(1)), int(m.group(2))))

    expected = set(labels.keys())
    missing = expected - cached_pairs
    n_extra = len(cached_pairs - expected)

    if not missing:
        extra_str = f", {n_extra} extra (OOM skips or eval traces)" if n_extra else ""
        return True, f"all {len(expected)} traces cached{extra_str}"
    else:
        return False, (
            f"{len(missing)}/{len(expected)} traces missing from cache "
            f"-- will collect missing ones"
        )


def bucket_files_by_question(
    files: List[Path],
    train_question_ids: List[int],
    val_question_ids: List[int],
    test_question_ids: List[int],
    seed: int = 42,
    max_traces_per_question: Optional[int] = None,
) -> Tuple[Dict[str, List[Path]], int]:
    """Bucket cache files into train/val/test splits by question ID.

    Files for question IDs not in any split (e.g. eval questions) are silently
    excluded. Returns (buckets, n_skipped) where n_skipped counts files whose
    names could not be parsed.

    Args:
        max_traces_per_question: If set, randomly keep at most this many files
            per question per split (seeded for reproducibility).
    """
    import random as _random

    train_set = set(train_question_ids)
    val_set = set(val_question_ids)
    test_set = set(test_question_ids)

    by_question: Dict[str, Dict[int, List[Path]]] = {"train": {}, "val": {}, "test": {}}
    skipped = 0
    for f in files:
        q_id = parse_cache_question_id(f.name)
        if q_id is None:
            skipped += 1
            continue
        if q_id in train_set:
            split = "train"
        elif q_id in val_set:
            split = "val"
        elif q_id in test_set:
            split = "test"
        else:
            continue  # eval question -- excluded
        by_question[split].setdefault(q_id, []).append(f)

    buckets: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for split, q_dict in by_question.items():
        for q_id, q_files in sorted(q_dict.items()):
            if max_traces_per_question is not None and len(q_files) > max_traces_per_question:
                rng = _random.Random(seed + q_id)
                q_files = rng.sample(q_files, max_traces_per_question)
            buckets[split].extend(q_files)

    return buckets, skipped
