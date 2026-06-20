"""Shared utilities for the attributions pipeline.

Consolidates data loading, thinking-token config, and answer extraction
that were previously spread across neurohike.core.data, shared.data_utils,
and shared.thinking_tokens.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Thinking-token configuration
# ---------------------------------------------------------------------------

def get_thinking_tokens(model_name: str) -> dict[str, Any]:
    """Get start/end thinking token metadata for a given model."""
    model_name_lower = model_name.lower()

    if "deepseek" in model_name_lower:
        return {
            "start_token": "<think>",
            "end_token": "</think>",
            "start_token_ids": None,
            "end_token_ids": None,
        }
    if "gpt-oss" in model_name_lower:
        return {
            "start_token": "<|channel|>analysis<|message|>",
            "end_token": "<|channel|>final<|message|>",
            "start_token_ids": [200005, 35644, 200008],
            "end_token_ids": [200007, 200006, 173781, 200005, 17196, 200008],
        }
    if "qwen" in model_name_lower:
        return {
            "start_token": "<think>",
            "end_token": "</think>",
            "start_token_ids": None,
            "end_token_ids": None,
        }
    if "gemma-4" in model_name_lower or "gemma4" in model_name_lower:
        return {
            "start_token": "<|channel>thought",
            "end_token": "<channel|>",
            "start_token_ids": None,
            "end_token_ids": None,
        }
    if "ministral" in model_name_lower or "mistral" in model_name_lower:
        return {
            "start_token": "[THINK]",
            "end_token": "[/THINK]",
            "start_token_ids": [34],
            "end_token_ids": [35],
        }
    return {
        "start_token": "<think>",
        "end_token": "</think>",
        "start_token_ids": None,
        "end_token_ids": None,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_data(data_name: str) -> list[tuple[str, int | str]]:
    """Load reasoning dataset.

    Args:
        data_name: Dataset identifier. Supported:
            - "gsm8k", "AIME2025", "aime_2024", "GSM-Plus-modified",
              "non-math-mmlu-pro", "ZebraLogic", "GPQA", "MATH-500"

    Returns:
        List of (question, answer) tuples.
    """

    if 'gsm8k'.casefold() in data_name.casefold():
        dataset = load_dataset("openai/gsm8k", "main", split="train")

        def extract_answer(answer_text: str) -> int:
            return int(
                answer_text.split("#### ")[-1]
                .strip()
                .replace(",", "")
            )

        return [
            (item["question"], extract_answer(item["answer"]))
            for item in dataset
        ]

    elif 'aime2025'.casefold() in data_name.casefold():
        dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        reasoning_dataset1 = [
            (item["question"] + instr, int(item["answer"]))
            for item in dataset1
        ]
        reasoning_dataset2 = [
            (item["question"] + instr, int(item["answer"].replace(r"^\circ", "")))
            for item in dataset2
        ]
        return reasoning_dataset1 + reasoning_dataset2

    elif 'aime_2024'.casefold() in data_name.casefold():
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

        reasoning_dataset = []
        for item in dataset:
            try:
                ans_str = str(item["answer"]).strip().replace(r"\\boxed{", "").replace("}", "")
                reasoning_dataset.append((item["problem"], int(ans_str)))
            except (ValueError, KeyError):
                continue
        return reasoning_dataset

    elif 'aime_2026'.casefold() in data_name.casefold():
        dataset = load_dataset("MathArena/aime_2026", split="train")

        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        reasoning_dataset = []
        for item in dataset:
            try:
                reasoning_dataset.append((item["problem"] + instr, int(item["answer"])))
            except (ValueError, KeyError):
                continue
        return reasoning_dataset

    elif 'GSM-Plus-modified'.casefold() in data_name.casefold():
        dataset = load_dataset(data_name, split="train")

        reasoning_dataset = []
        for item in dataset:
            try:
                reasoning_dataset.append((item["question"], str(item["answer"]).strip()))
            except (ValueError, KeyError):
                continue
        return reasoning_dataset

    elif 'non-math-mmlu-pro'.casefold() in data_name.casefold():
        dataset = load_dataset('TIGER-Lab/MMLU-Pro', split="validation")
        filtered_dataset = [ele for ele in dataset if ele['category'] != 'math']

        reasoning_dataset = []
        for item in filtered_dataset:
            try:
                question = item['question']
                options = item['options']
                options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                instr = "\nAnswer with the letter corresponding to the correct option."
                full_question = question + "\n" + options_str + instr
                reasoning_dataset.append((full_question, item['answer']))
            except (ValueError, KeyError, IndexError):
                continue
        return reasoning_dataset

    elif 'ZebraLogic'.casefold() in data_name.casefold():
        dataset = load_dataset('WildEval/ZebraLogic', 'mc_mode', split="test")

        seed = 12
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), size=50, replace=False)
        dataset = dataset.select(indices)

        reasoning_dataset = []
        for item in dataset:
            try:
                puzzle = item['puzzle']
                question = item['question']
                choices = item['choices']
                instr = "\nAnswer with one of the following options in a \\boxed{} environment:"
                full_question = puzzle + "\n" + question + "\n" + instr + "\n".join(choices)
                reasoning_dataset.append((full_question, item['answer']))
            except (ValueError, KeyError, IndexError):
                continue
        return reasoning_dataset

    elif 'GPQA'.casefold() in data_name.casefold():
        dataset = load_dataset('fingertap/GPQA-Diamond', split='test')

        reasoning_dataset = []
        instr = "\nAnswer with the letter corresponding to the correct option in a \\boxed{} environment."

        for item in dataset:
            question = item['question']
            answer = item['answer']
            reasoning_dataset.append(
                (question + instr, answer)
            )
        return reasoning_dataset

    elif 'MATH-500'.casefold() in data_name.casefold():
        dataset = load_dataset('HuggingFaceH4/MATH-500', split="test")

        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        reasoning_dataset = []
        for item in dataset:
            reasoning_dataset.append((item["problem"] + instr, str(item["answer"])))
        return reasoning_dataset

    elif 'bigbenchhard'.casefold() in data_name.casefold() or 'bbh' in data_name.casefold():
        import requests

        all_configs = [
            'tracking_shuffled_objects_seven_objects',
            'salient_translation_error_detection',
            'tracking_shuffled_objects_three_objects',
            'geometric_shapes',
            'object_counting',
            'word_sorting',
            'logical_deduction_five_objects',
            'hyperbaton',
            'sports_understanding',
            'logical_deduction_seven_objects',
            'multistep_arithmetic_two',
            'ruin_names',
            'causal_judgement',
            'logical_deduction_three_objects',
            'formal_fallacies',
            'snarks',
            'boolean_expressions',
            'reasoning_about_colored_objects',
            'dyck_languages',
            'navigate',
            'disambiguation_qa',
            'temporal_sequences',
            'web_of_lies',
            'tracking_shuffled_objects_five_objects',
            'penguins_in_a_table',
            'movie_recommendation',
            'date_understanding',
        ]

        full_dataset = []
        for config in all_configs:
            url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{config}.json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                full_dataset.extend(data['examples'][:20])
            else:
                print(f"Failed to fetch data for config: {config} when downloading from {url}")

        random.seed(12)
        random.shuffle(full_dataset)

        reasoning_dataset = []
        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        for item in full_dataset:
            question = item['input']
            answer = item['target']
            reasoning_dataset.append((question + instr, answer))
        return reasoning_dataset

    elif 'humanevalplus' in data_name.casefold():
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        reasoning_dataset = []
        for item in dataset:
            reasoning_dataset.append((item["prompt"], item["canonical_solution"]))
        return reasoning_dataset

    elif 'live_code_bench' in data_name.casefold():
        return [(item["prompt"], "") for item in _load_live_code_bench_items()]

    else:
        raise ValueError(f"Unknown dataset {data_name}")


def extract_boxed_answer(text: str, first: bool = False) -> str:
    """Extract content from a balanced \\boxed{} occurrence in the string."""
    if first:
        search_from = 0
        while True:
            start = text.find('\\boxed{', search_from)
            if start == -1:
                return ''

            answer = _extract_boxed_answer_at(text, start)
            if answer.strip():
                return answer
            search_from = start + 7

    start = text.rfind('\\boxed{')
    if start == -1:
        return ''

    return _extract_boxed_answer_at(text, start)


def _extract_boxed_answer_at(text: str, start: int) -> str:
    """Extract balanced boxed content starting at a known ``\\boxed{`` index."""

    start_content = start + 7
    brace_count = 1
    i = start_content

    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start_content:i-1]

    return ''


# ---------------------------------------------------------------------------
# Code benchmark helpers
# ---------------------------------------------------------------------------

def is_code_benchmark(data_name: str) -> bool:
    """Check whether data_name refers to a code-generation benchmark."""
    name = data_name.casefold()
    return 'humanevalplus' in name or 'live_code_bench' in name


def get_answer_domain(data_name: str) -> str:
    """Return the answer-verification domain for a dataset.

    Kept alongside ``get_data``/``get_answer_suffix`` so dataset-specific answer
    semantics stay in one small policy layer.
    """
    name = data_name.casefold()
    if is_code_benchmark(data_name):
        return "code"
    if any(key in name for key in ("mmlu", "zebralogic", "gpqa")):
        return "mcq"
    return "math"


def get_answer_suffix(data_name: str) -> str:
    """Return the answer suffix appropriate for the dataset.

    Math/reasoning benchmarks use the \\boxed{} prompt; code benchmarks
    use a plain "this is the solution:" prompt so the model produces code.
    """
    if is_code_benchmark(data_name):
        return "Therefore, this is the solution:\n```python\n"
    return r"Therefore, the final answer is \boxed{"


def _resolve_data_path(*parts: str) -> Path:
    return Path("data", *parts)


_LIVE_CODE_BENCH_PATH = _resolve_data_path("live_code_bench", "test6.jsonl")

_LIVE_CODE_BENCH_INSTRUCTIONS = (
    "You are solving a competitive-programming problem. Follow these rules strictly so your "
    "answer can be graded automatically:\n"
    "1. Write a single complete Python 3 program — do NOT use C++, Java, or any other language.\n"
    "2. The program must read its input from standard input (stdin) and print the answer to "
    "standard output (stdout). Do not wrap the logic in a function unless you also call it at "
    "the bottom of the file.\n"
    "3. Return exactly ONE fenced code block in your final answer, tagged as ```python. Do not "
    "include alternative solutions, partial snippets, or extra code blocks anywhere in the "
    "response.\n"
    "4. The code block must be self-contained and runnable as-is with `python solution.py < input`. "
    "No placeholders, no `TODO`s, no interactive prompts, no reading from files.\n"
    "5. Print only what the problem specifies — no debug prints, no trailing explanation lines "
    "inside the program's output.\n\n"
    "Problem:\n"
)


def _load_live_code_bench_items() -> list[dict]:
    """Load, filter, and decode the live_code_bench source JSONL.

    Keeps only `difficulty == "hard"` items whose tests are all `testtype == "stdin"`.
    Returns a list of `{"task_id", "prompt", "tests"}` dicts, in source order.

    The same helper backs both `get_data()` and `get_code_benchmark_metadata()` so
    positional indexing is guaranteed to match.
    """
    import base64
    import pickle
    import zlib

    if not _LIVE_CODE_BENCH_PATH.exists():
        raise FileNotFoundError(
            f"live_code_bench source not found at {_LIVE_CODE_BENCH_PATH}"
        )

    items: list[dict] = []
    with open(_LIVE_CODE_BENCH_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if raw.get("difficulty") != "medium":
                continue
            decoded = base64.b64decode(raw["private_test_cases"])
            recovered = json.loads(pickle.loads(zlib.decompress(decoded)))
            if any(t.get("testtype") != "stdin" for t in recovered):
                continue
            items.append({
                "task_id": raw["question_id"],
                "prompt": _LIVE_CODE_BENCH_INSTRUCTIONS + raw["question_content"],
                "tests": recovered,
            })
    return items


def get_code_benchmark_metadata(data_name: str) -> Optional[list[dict]]:
    """Return per-question metadata for code benchmarks, or None otherwise.

    - humanevalplus: list of {task_id, entry_point, prompt}
    - live_code_bench: list of {task_id, tests}
    """
    if not is_code_benchmark(data_name):
        return None
    name = data_name.casefold()
    if 'live_code_bench' in name:
        return [
            {"task_id": item["task_id"], "tests": item["tests"]}
            for item in _load_live_code_bench_items()
        ]
    dataset = load_dataset("evalplus/humanevalplus", split="test")
    return [
        {
            "task_id": item["task_id"],
            "entry_point": item["entry_point"],
            "prompt": item["prompt"],
        }
        for item in dataset
    ]


def extract_code_answer(text: str, model_name: str) -> str:
    """Extract clean code from a model generation, stripping thinking tokens.

    Uses the appropriate end-of-thinking marker based on model_name.
    Discards generations that contain excessive repeated lines (degenerate output).
    """
    end_pattern = get_thinking_tokens(model_name)["end_token"]

    result = text.strip()

    think_index = result.find(end_pattern)
    if think_index != -1:
        result = result[think_index + len(end_pattern):].strip()
    else:
        result = ""  # no end-thinking marker → discard

    # Reject degenerate output with excessive repeated *non-blank* lines.
    # (Blank lines naturally pile up in long code/prose and are not pathology.)
    if result:
        counts: dict[str, int] = {}
        for line in result.split("\n"):
            if not line.strip():
                continue
            counts[line] = counts.get(line, 0) + 1
            if counts[line] > 60:
                return "-"

    return result


def evaluate_code_batch(
    samples: list[dict],
    tmp_dir: Path,
    parallel: int = 6,
) -> dict[str, list[dict]]:
    """Batch-evaluate code solutions via EvalPlus.

    Args:
        samples: list of {"task_id": str, "solution": str} dicts.
        tmp_dir: directory for intermediate files (samples.jsonl, sanitized, results).
        parallel: number of parallel workers for EvalPlus.

    Returns:
        Dict mapping task_id → list of per-solution result dicts.
        Each result dict has at least "plus_status" (str: "pass" or "fail").
    """
    from evalplus.sanitize import script as sanitize_script
    from evalplus.evaluate import evaluate

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # EvalPlus's evaluate() asserts every benchmark task_id is present in the
    # samples. When called on a per-question subset we only have one (or a few),
    # so pad with a single dummy entry per missing task_id. These extras fail
    # the tests but are ignored by the caller (results are keyed by task_id).
    present_task_ids = {s["task_id"] for s in samples}
    all_task_ids = {
        item["task_id"]
        for item in load_dataset("evalplus/humanevalplus", split="test")
    }
    missing = sorted(all_task_ids - present_task_ids)

    # Write samples.jsonl
    samples_file = tmp_dir / "samples.jsonl"
    with open(samples_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
        for task_id in missing:
            f.write(json.dumps({"task_id": task_id, "solution": ""}) + "\n")

    # Sanitize (always re-run so sanitized file stays in sync with samples.jsonl)
    sanitized_file = samples_file.with_name(samples_file.stem + "-sanitized.jsonl")
    sanitize_script(samples=str(samples_file))

    # Evaluate
    evaluate(
        dataset="humaneval",
        samples=str(sanitized_file),
        parallel=parallel,
    )

    # Parse results
    results_file = sanitized_file.with_name(
        sanitized_file.stem + "_eval_results.json"
    )
    if not results_file.exists():
        print(f"WARNING: EvalPlus results file not found at {results_file}")
        return {}

    with open(results_file) as f:
        raw = json.load(f)

    return raw.get("eval", {})


def _extract_program_from_text(text: str) -> str:
    """Peel a runnable Python program out of a model response.

    live_code_bench has no evalplus-style sanitizer, so we handle the common
    gpt-oss / qwen shape: prose interleaved with one or more ```python ... ```
    (or bare ``` ... ```) fenced blocks. We keep the longest fenced block
    whose content looks like Python (as a proxy: has `def`, `import`, `for`,
    `while`, `print`, or an assignment). If no fence is present we return the
    text as-is.
    """
    import re

    if not text:
        return ""

    # Greedy-match fenced blocks; optional language tag.
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if not blocks:
        return text

    def _looks_like_python(block: str) -> bool:
        stripped = block.strip()
        if not stripped:
            return False
        keywords = ("def ", "import ", "from ", "for ", "while ", "print(", "class ", "if __name__")
        if any(k in stripped for k in keywords):
            return True
        # Assignment heuristic: a line with `=` that isn't just `==`.
        for line in stripped.splitlines():
            s = line.strip()
            if "=" in s and "==" not in s and not s.startswith("#"):
                return True
        return False

    candidates = [b for b in blocks if _looks_like_python(b)] or blocks
    return max(candidates, key=len).strip()


def _run_stdin_solution(
    solution_path: Path,
    tests: list[dict],
    timeout: float,
) -> tuple[bool, str]:
    """Run a Python program against stdin/stdout tests.

    Returns (passed_all, reason). `reason` is empty on success, else a short
    description of the first failure (for logging).
    """
    import subprocess
    import sys

    for i, t in enumerate(tests):
        assert t.get("testtype") == "stdin", (
            f"Non-stdin test leaked through filter: {t.get('testtype')!r}"
        )
        expected = (t.get("output") or "").strip()
        try:
            proc = subprocess.run(
                [sys.executable, str(solution_path)],
                input=t.get("input", ""),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # means that the code ran but took too long (e.g. infinite loop, excessive computation, etc.)
            return False, f"test {i}: timeout, exceeded {timeout} seconds"
        except Exception as exc:  # pragma: no cover – defensive
            # means that the code didn't run successfully (e.g. syntax error, import error, etc.)
            return False, f"test {i}: exec error ({exc!r})"

        if proc.returncode != 0:
            # means that the code didn't run successfully
            return False, f"test {i}: non-zero exit ({proc.returncode}), stderr: {proc.stderr.strip()!r}"

        actual = (proc.stdout or "").strip()
        if actual != expected:
            # means that the code ran but produced wrong answer
            return False, f"test {i}: stdout mismatch, expected {expected!r}, got {actual!r}"

    return True, ""


def evaluate_live_code_batch(
    samples: list[dict],
    tmp_dir: Path,
    metadata_by_task_id: dict[str, list[dict]],
    parallel: int = 6,
    per_test_timeout: float = 6.0,
) -> dict[str, list[dict]]:
    """Batch-evaluate live_code_bench solutions via subprocess stdin/stdout diff.

    Args:
        samples: list of {"task_id": str, "solution": str} dicts, in queue order.
        tmp_dir: directory for intermediate solution files.
        metadata_by_task_id: maps task_id → list of test dicts
            (each {"input", "output", "testtype": "stdin"}).
        parallel: number of concurrent workers.
        per_test_timeout: per-test subprocess timeout in seconds.

    Returns:
        Dict mapping task_id → list of per-solution result dicts (in input order).
        Each result dict has at least "plus_status" (str: "pass" or "fail").
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Per-task input-order position so results can be reassembled deterministically.
    per_task_positions: dict[str, list[int]] = {}
    jobs: list[tuple[int, str, str]] = []  # (global_idx, task_id, solution)
    for gi, s in enumerate(samples):
        task_id = s["task_id"]
        per_task_positions.setdefault(task_id, []).append(gi)
        jobs.append((gi, task_id, s.get("solution", "") or ""))

    results_by_global: dict[int, dict] = {}

    def _run_one(gi: int, task_id: str, solution: str) -> tuple[int, dict]:
        if not solution.strip():
            return gi, {"plus_status": "fail", "reason": "empty solution"}
        program = _extract_program_from_text(solution)
        if not program.strip():
            return gi, {"plus_status": "fail", "reason": "no code block"}
        tests = metadata_by_task_id.get(task_id)
        if not tests:
            return gi, {"plus_status": "fail", "reason": "no tests for task_id"}
        sol_file = tmp_dir / f"sol_{gi:06d}.py"
        sol_file.write_text(program)
        try:
            passed, reason = _run_stdin_solution(sol_file, tests, per_test_timeout)
        finally:
            try:
                sol_file.unlink()
            except OSError:
                pass
        return gi, {"plus_status": "pass" if passed else "fail", "reason": reason}

    first_fail_logged = False
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = [pool.submit(_run_one, gi, tid, sol) for gi, tid, sol in jobs]
        for fut in as_completed(futures):
            gi, entry = fut.result()
            results_by_global[gi] = entry
            if not first_fail_logged and entry["plus_status"] == "fail" and entry.get("reason"):
                print(f"    live_code_bench: first failure – {entry['reason']}")
                first_fail_logged = True

    out: dict[str, list[dict]] = {}
    for task_id, positions in per_task_positions.items():
        out[task_id] = [results_by_global[p] for p in positions]
    return out


# ---------------------------------------------------------------------------
# Shared code-eval glue (used by both vllm_traces_generator.py and
# probe_compression_exps.py to keep grading + per-trace annotation uniform).
# ---------------------------------------------------------------------------

def run_code_eval(
    samples: list[dict],
    out_dir: Path,
    is_live_code: bool,
    tests_by_task: Optional[dict[str, list[dict]]] = None,
) -> dict[str, list[dict]]:
    """Dispatch to the right batch grader.

    `samples` = [{"task_id": str, "solution": str}, ...].
    Returns the raw `dict[task_id → list[per-solution-result]]` produced by
    the underlying evaluator. Each result has at least `plus_status`.
    """
    if is_live_code:
        return evaluate_live_code_batch(samples, out_dir, tests_by_task or {})
    return evaluate_code_batch(samples, out_dir)


def parse_code_eval_entry(entry: dict) -> tuple[bool, Optional[str]]:
    """Read one per-solution dict from `run_code_eval`.

    Returns `(is_pass, fail_reason)`. Pass → (True, None);
    Fail → (False, entry.get("reason") or "fail").
    """
    if entry.get("plus_status") == "pass":
        return True, None
    return False, (entry.get("reason") or "fail")


def assemble_solution_for_eval(answer_suffix: str, generated: str) -> str:
    """Prepend the answer-suffix opener (e.g. "```python\\n" or "\\boxed{")
    to a model continuation so extractors / EvalPlus see the canonical
    delimiters that were tokenized into the prompt rather than emitted by
    the model. Empty generations stay empty (still graded as 'fail').
    """
    if not generated:
        return generated
    return answer_suffix + generated


# ---------------------------------------------------------------------------
# Reasoning traces
# ---------------------------------------------------------------------------

def get_reasoning_traces(
    model_name: str,
    data_name: str,
) -> list[dict[str, Any]]:
    """Load pre-generated reasoning traces.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-4B").
        data_name: Dataset name (e.g., "openai/gsm8k").

    Returns:
        List of trace dictionaries containing input_text, traces,
        extracted_answers, etc.
    """
    relative = Path(
        data_name.split('/')[-1],
        f"{model_name.split('/')[-1]}_teacher_traces.json"
    )
    traces_path = Path("data") / relative
    if not traces_path.exists():
        raise FileNotFoundError(f"Reasoning traces file not found: {traces_path}")

    with open(traces_path, 'r') as f:
        reasoning_traces: list[dict[str, str]] = json.load(f)

    return reasoning_traces


# ---------------------------------------------------------------------------
# Question-level split for probe training vs compression eval
# ---------------------------------------------------------------------------

def generate_question_split(
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    save_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Generate a deterministic question-level split.

    Partitions question indices into probe_train / probe_val / probe_test / eval
    using a seeded permutation. If save_path exists, loads and verifies it matches.
    """
    assert n_train + n_val + n_test <= n_total, (
        f"train({n_train}) + val({n_val}) + test({n_test}) exceeds n_total({n_total})"
    )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total).tolist()

    split = {
        "seed": seed,
        "n_total": n_total,
        "probe_train": sorted(perm[:n_train]),
        "probe_val": sorted(perm[n_train:n_train + n_val]),
        "probe_test": sorted(perm[n_train + n_val:n_train + n_val + n_test]),
        "eval": sorted(perm[n_train + n_val + n_test:]),
    }

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.exists():
            existing = load_question_split(save_path)
            for key in ("probe_train", "probe_val", "probe_test", "eval"):
                if existing[key] != split[key]:
                    raise ValueError(
                        f"Existing split at {save_path} (seed={existing.get('seed')}) "
                        f"does not match generated split (seed={seed}). "
                        f"Delete the file or use a matching seed."
                    )
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(split, f, indent=2)

    return split


def load_question_split(path: str | Path) -> dict[str, Any]:
    """Load and validate a question split JSON file."""
    path = Path(path)
    with open(path) as f:
        split = json.load(f)

    required_keys = {"probe_train", "probe_val", "probe_test", "eval"}
    missing = required_keys - set(split.keys())
    if missing:
        raise ValueError(f"Split file {path} missing keys: {missing}")

    all_ids = (
        set(split["probe_train"])
        | set(split["probe_val"])
        | set(split["probe_test"])
        | set(split["eval"])
    )
    n_total = split.get("n_total", len(all_ids))
    if all_ids != set(range(n_total)):
        raise ValueError(f"Split file {path}: IDs don't cover range(0, {n_total})")
    if sum(len(split[k]) for k in required_keys) != n_total:
        raise ValueError(f"Split file {path}: partitions overlap")

    return split
