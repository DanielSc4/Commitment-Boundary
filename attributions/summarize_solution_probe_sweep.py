#!/usr/bin/env python3
"""Summarize safety-first solution-probe sweep results."""

import csv
import json
from pathlib import Path
from typing import Optional

import fire


def _get_class_metric(results: dict, class_id: int, name: str) -> Optional[float]:
    for item in results.get("test_metrics", {}).get("per_class", []):
        if int(item.get("class_id", -1)) == class_id:
            return item.get(name)
    return None


def _boundary(results: dict, key: str, name: str) -> Optional[float]:
    metrics = results.get("class2_boundary_metrics", {})
    block = metrics.get(key)
    if block is None:
        block = metrics.get(f"early_exit_{key.replace('=', '')}")
    return None if block is None else block.get(name)


def _rank_score(row: dict, min_detect_rate: float) -> float:
    early = row.get("k2_early_fire_rate")
    detect = row.get("k2_detect_rate")
    saved = row.get("k2_mean_saved_fraction_all")
    precision = row.get("class2_precision")
    if early is None or detect is None:
        return -1e9
    detect_penalty = max(0.0, min_detect_rate - detect)
    return (
        -10.0 * early
        - 5.0 * detect_penalty
        + 1.5 * detect
        + 0.75 * (saved or 0.0)
        + 0.5 * (precision or 0.0)
    )


def main(
    sweep_dir: str,
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
    min_detect_rate: float = 0.80,
):
    root = Path(sweep_dir)
    files = sorted(root.glob("**/solution_probe_results_L*.json"))
    rows = []
    for path in files:
        with open(path) as f:
            results = json.load(f)
        hparams = results.get("hyperparams", {})
        row = {
            "result_path": str(path),
            "output_dir": str(path.parent),
            "model": results.get("model"),
            "data_name": results.get("data_name"),
            "layer": results.get("layer"),
            "sentence_aggregation": hparams.get("sentence_aggregation"),
            "max_probe_input_tokens": hparams.get("max_probe_input_tokens"),
            "batch_size": hparams.get("batch_size"),
            "seed": hparams.get("seed"),
            "accuracy": results.get("test_metrics", {}).get("accuracy"),
            "macro_f1": results.get("test_metrics", {}).get("macro_f1"),
            "class2_precision": _get_class_metric(results, 2, "precision"),
            "class2_recall": _get_class_metric(results, 2, "recall"),
            "class2_f1": _get_class_metric(results, 2, "f1"),
            "class1_f1": _get_class_metric(results, 1, "f1"),
            "k1_early_fire_rate": _boundary(results, "k=1", "early_fire_rate"),
            "k1_detect_rate": _boundary(results, "k=1", "detect_rate"),
            "k1_mean_saved_fraction_all": _boundary(results, "k=1", "mean_saved_fraction_all"),
            "k2_early_fire_rate": _boundary(results, "k=2", "early_fire_rate"),
            "k2_detect_rate": _boundary(results, "k=2", "detect_rate"),
            "k2_miss_rate": _boundary(results, "k=2", "miss_rate"),
            "k2_mean_saved_fraction_all": _boundary(results, "k=2", "mean_saved_fraction_all"),
            "k3_early_fire_rate": _boundary(results, "k=3", "early_fire_rate"),
            "k3_detect_rate": _boundary(results, "k=3", "detect_rate"),
            "k3_miss_rate": _boundary(results, "k=3", "miss_rate"),
            "k3_mean_saved_fraction_all": _boundary(results, "k=3", "mean_saved_fraction_all"),
        }
        row["rank_score"] = _rank_score(row, min_detect_rate=min_detect_rate)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["rank_score"],
            -(row.get("k2_early_fire_rate") or 1.0),
            row.get("k2_detect_rate") or 0.0,
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    if output_csv is None:
        output_csv = str(root / "sweep_summary.csv")
    if output_json is None:
        output_json = str(root / "sweep_summary.json")

    out_csv = Path(output_csv)
    out_json = Path(output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote ranked JSON to {out_json}")
    for row in rows[:10]:
        print(
            f"#{row['rank']:02d} L{row['layer']} {row['sentence_aggregation']} "
            f"window={row['max_probe_input_tokens']} "
            f"k2_early={row.get('k2_early_fire_rate')} "
            f"k2_detect={row.get('k2_detect_rate')} "
            f"saved={row.get('k2_mean_saved_fraction_all')} "
            f"p2={row.get('class2_precision')} "
            f"dir={row['output_dir']}"
        )


if __name__ == "__main__":
    fire.Fire(main)
