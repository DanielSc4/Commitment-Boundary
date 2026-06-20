import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path (needed when running inside Singularity)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from attributions.utils import (
    get_data, extract_boxed_answer, is_code_benchmark,
    extract_code_answer, get_code_benchmark_metadata,
    get_thinking_tokens, get_answer_suffix,
    run_code_eval, parse_code_eval_entry,
)


def _run_pass(
    *,
    llm,
    sampling_params,
    prompts,
    reasoning_dataset,
    tokenizer,
    data_name: str,
    model_name: str,
    code_meta,
    is_live_code: bool,
    num_out: int,
    batch_size: int,
    resume: bool,
    output_dir: Path,
    final_file: Path,
    batch_dir: Path,
    readable_file: Path,
    pass_label: str,
):
    """Run one generation pass (CoT or no-CoT), save batches, run sanity eval.

    `prompts[i]` is either a str or a dict {"prompt_token_ids": [...]} — vLLM
    accepts both transparently via `llm.generate`.
    """

    # Determine which indices to process (support resume on a per-pass basis).
    processed_indices: set = set()
    if resume and final_file.exists():
        print(f"[{pass_label}] Checking existing progress in {final_file}...")
        try:
            with open(final_file, "r") as f:
                existing_data = json.load(f)
                processed_indices = {item["input_text"] for item in existing_data}
                print(f"[{pass_label}] Found {len(processed_indices)} already processed examples. Resuming...")
        except Exception as e:
            print(f"[{pass_label}] Could not read existing file (maybe corrupted): {e}. Starting from scratch.")

    indices_to_process = [
        i for i, (question, _) in enumerate(reasoning_dataset)
        if question not in processed_indices
    ]

    if not indices_to_process:
        print(f"[{pass_label}] All examples already processed!")
        # Still ensure all_results is loaded for the sanity-check section
        with open(final_file, "r") as f:
            all_results = json.load(f)
    else:
        print(f"[{pass_label}] Will process {len(indices_to_process)} remaining examples in batches of {batch_size}")

        if not final_file.exists():
            with open(final_file, "w") as f:
                json.dump([], f)

        batch_dir.mkdir(exist_ok=True, parents=True)

        for batch_start in tqdm(range(0, len(indices_to_process), batch_size), desc=f"Batches [{pass_label}]"):

            batch_indices = indices_to_process[batch_start:batch_start + batch_size]
            batch_prompts = [prompts[i] for i in batch_indices]
            batch_questions = [reasoning_dataset[i][0] for i in batch_indices]
            batch_gt_answers = [reasoning_dataset[i][1] for i in batch_indices]

            print(f"\n[{pass_label}] Generating batch {batch_start // batch_size + 1} "
                  f"({len(batch_prompts)} prompts)...")

            outputs = llm.generate(batch_prompts, sampling_params)

            batch_results = []
            for idx_in_batch, output in enumerate(outputs):
                global_idx = batch_indices[idx_in_batch]
                question = batch_questions[idx_in_batch]
                gt_answer = batch_gt_answers[idx_in_batch]

                traces = [output.outputs[x].text for x in range(len(output.outputs))]
                traces_tokens = [list(output.outputs[x].token_ids) for x in range(len(output.outputs))]
                # Merge prompt + generation before extraction so the end-of-thinking
                # marker (and any answer-suffix opener like ```python or \boxed{) is
                # always present, regardless of whether we're in CoT or no-CoT mode.
                prompt_text = tokenizer.decode(output.prompt_token_ids, skip_special_tokens=False)
                if is_code_benchmark(data_name):
                    extracted = [extract_code_answer(prompt_text + t, model_name) for t in traces]
                elif "ministral" in model_name.lower() or "mistral" in model_name.lower():
                    extracted = [extract_boxed_answer(prompt_text + t, first=True) for t in traces]
                else:
                    extracted = [extract_boxed_answer(prompt_text + t) for t in traces]

                # Compute top-k entropy for each token in each trace
                traces_entropy = []
                for completion in output.outputs:
                    token_entropies = []
                    for step in completion.logprobs:
                        log_probs = np.array([v.logprob for v in step.values()])
                        probs = np.exp(log_probs)
                        probs = probs / probs.sum()  # renormalize over top-k
                        entropy = -np.sum(probs * np.log(probs + 1e-12))
                        token_entropies.append(float(entropy))
                    traces_entropy.append(token_entropies)

                result = {
                    "input_text": question,
                    "prompt_tokens": list(output.prompt_token_ids),
                    "traces": traces,
                    "traces_tokens": traces_tokens,
                    "traces_entropy": traces_entropy,
                    "extracted_answers": extracted,
                    "GT_answer": gt_answer,
                }
                if code_meta is not None:
                    result["task_id"] = code_meta[global_idx]["task_id"]
                    if is_live_code:
                        result["tests"] = code_meta[global_idx]["tests"]
                    else:
                        result["entry_point"] = code_meta[global_idx]["entry_point"]
                batch_results.append(result)

            # save this batch separately (safety)
            batch_file = batch_dir / f"batch_{batch_start:06d}_{batch_start + len(batch_results):06d}.json"
            with open(batch_file, "w") as f:
                json.dump(batch_results, f, indent=4, ensure_ascii=False)

            # append to the main file
            with open(final_file, "r") as f:
                all_results = json.load(f)
            all_results.extend(batch_results)
            with open(final_file, "w") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

            print(f"[{pass_label}] Batch saved. Total processed: {len(all_results)}")

    # ── Sanity-check metrics ──────────────────────────────────────────
    # Also annotates each item in `all_results` with per-trace pass/fail info
    # so the saved JSON can be quickly inspected (and failing solutions copied
    # out to reproduce locally).
    if is_code_benchmark(data_name):
        label = "live_code_bench stdin" if is_live_code else "EvalPlus"
        print(f"\n[{pass_label}] Running {label} sanity-check evaluation...")
        samples = []
        task_id_to_idx = {}
        for item in all_results:
            task_id = item.get("task_id", "")
            if task_id not in task_id_to_idx:
                task_id_to_idx[task_id] = []
            for code in item["extracted_answers"]:
                task_id_to_idx[task_id].append(len(samples))
                samples.append({"task_id": task_id, "solution": code})

        if samples:
            tmp_dir = output_dir / (
                f"live_code_bench_sanity_{pass_label}" if is_live_code
                else f"evalplus_sanity_{pass_label}"
            )
            tests_by_task = (
                {it["task_id"]: it["tests"] for it in all_results} if is_live_code else None
            )
            eval_results = run_code_eval(
                samples, tmp_dir, is_live_code=is_live_code, tests_by_task=tests_by_task,
            )
            n_evaluated = 0
            pass_at_k_hits = 0
            pass_rates = []
            for item in all_results:
                task_id = item.get("task_id", "")
                entries = eval_results.get(task_id, [])
                # Annotate per-trace correctness and (when present) failure reason.
                parsed = [parse_code_eval_entry(e) for e in entries]
                item["trace_correct"] = [is_pass for is_pass, _ in parsed]
                item["trace_fail_reason"] = [reason for _, reason in parsed]
                if not entries:
                    continue
                n_evaluated += 1
                correct = [1 if c else 0 for c in item["trace_correct"]]
                pass_at_k_hits += int(any(c == 1 for c in correct))
                pass_rates.append(sum(correct) / len(correct))

            if n_evaluated > 0:
                avg_pass_at_k = pass_at_k_hits / n_evaluated
                avg_pass_rate = sum(pass_rates) / n_evaluated
                print(f"\n{'='*50}")
                print(f"[{pass_label}] {label} sanity-check (n={n_evaluated}, k={num_out}):")
                print(f"  avg pass@{num_out}:  {avg_pass_at_k:.4f}")
                print(f"  avg pass_rate: {avg_pass_rate:.4f}")
                print(f"{'='*50}")
            else:
                print(f"\n[{pass_label}] WARNING: {label} returned no results.")
        else:
            print(f"\n[{pass_label}] WARNING: No code extractions to evaluate.")
    else:
        pass_at_k_hits = 0
        pass_rates = []
        n_evaluated = 0

        for item in all_results:
            extracted = item["extracted_answers"]
            gt = str(item["GT_answer"]).strip()
            # Annotate per-trace correctness (extracted boxed answer == GT).
            item["trace_correct"] = [str(e).strip() == gt for e in extracted]
            if not extracted:
                continue
            n_evaluated += 1
            correct = [1 if c else 0 for c in item["trace_correct"]]
            pass_at_k_hits += int(any(c == 1 for c in correct))
            pass_rates.append(sum(correct) / len(correct))

        if n_evaluated > 0:
            avg_pass_at_k = pass_at_k_hits / n_evaluated
            avg_pass_rate = sum(pass_rates) / n_evaluated
            print(f"\n{'='*50}")
            print(f"[{pass_label}] Sanity-check metrics (n={n_evaluated}, k={num_out}):")
            print(f"  avg pass@{num_out}:  {avg_pass_at_k:.4f}")
            print(f"  avg pass_rate: {avg_pass_rate:.4f}")
            print(f"{'='*50}")
        else:
            print(f"\n[{pass_label}] WARNING: No examples had extracted answers — cannot compute metrics.")

    # Re-write the final JSON so the per-trace pass/fail annotations land on disk.
    with open(final_file, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Save a human-readable JSONL (entropy lengths instead of raw lists).
    # Written after eval so it can include `trace_correct` for quick scanning.
    with open(readable_file, "w") as f:
        for item in all_results:
            readable = {
                "input_text": item["input_text"],
                "prompt_tokens_len": len(item["prompt_tokens"]),
                "num_traces": len(item["traces"]),
                "traces_tokens_len": [len(t) for t in item["traces_tokens"]],
                "traces_entropy_len": [len(e) for e in item["traces_entropy"]],
                "extracted_answers": item["extracted_answers"],
                "GT_answer": item["GT_answer"],
                "trace_correct": item.get("trace_correct", []),
            }
            if "trace_fail_reason" in item:
                readable["trace_fail_reason"] = item["trace_fail_reason"]
            f.write(json.dumps(readable, ensure_ascii=False) + "\n")

    print(f"\n[{pass_label}] Done! Final results: {final_file}")
    print(f"[{pass_label}] Readable summary: {readable_file}")
    print(f"[{pass_label}] Batches in: {batch_dir}")


def main(
    model_name: str = "openai/gpt-oss-20b",
    data_name: str = "HuggingFaceH4/MATH-500",
    num_out: int = 16,
    batch_size: int = 100,
    resume: bool = True,          # automatically resume if partial results exist
    quantization: Optional[str] = None,  # quantization mode: "awq", "gptq", "squeezellm", "fp8", etc.
    no_cot: bool = True,         # if True, additionally run a no-CoT pass with [start][end][suffix]
    tensor_parallel_size: int = 1,
    max_questions: Optional[int] = None,
    **gen_kwargs,
):
    print(f"Loading dataset: {data_name}")
    reasoning_dataset = get_data(data_name)
    if max_questions is not None:
        reasoning_dataset = reasoning_dataset[:max_questions]
    print(f"Loaded {len(reasoning_dataset)} examples")

    # For code benchmarks, load per-question metadata (task_id, entry_point / tests)
    code_meta = get_code_benchmark_metadata(data_name)
    if code_meta is not None and max_questions is not None:
        code_meta = code_meta[:max_questions]
    is_live_code = 'live_code_bench' in data_name.casefold()

    print(f"Loading model: {model_name}")
    if quantization:
        print(f"Using quantization: {quantization}")
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
    )
    if "ministral" in model_name.lower() or "mistral" in model_name.lower():
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    default_gen_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 16_384,
    }
    gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
    top_k_entropy = 20
    sampling_params = SamplingParams(
        n=num_out, logprobs=top_k_entropy,
        skip_special_tokens=False,  # keep special tokens so end-of-thinking markers are canonical
        **gen_kwargs,
    )

    dataset_short_name = data_name.split("/")[-1]
    model_short_name = model_name.split("/")[-1]
    output_dir = Path("./data") / dataset_short_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── CoT prompts (current behavior) ────────────────────────────────
    is_gemma4 = "gemma-4" in model_name.lower() or "gemma4" in model_name.lower()
    cot_prompts = []
    for question, _ in reasoning_dataset:
        messages = [{"role": "user", "content": question}]
        extra_kwargs = {"enable_thinking": True} if is_gemma4 else {}
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **extra_kwargs,
        )
        cot_prompts.append(prompt)

    # ── Pass 1: CoT ───────────────────────────────────────────────────
    _run_pass(
        llm=llm,
        sampling_params=sampling_params,
        prompts=cot_prompts,
        reasoning_dataset=reasoning_dataset,
        tokenizer=tokenizer,
        data_name=data_name,
        model_name=model_name,
        code_meta=code_meta,
        is_live_code=is_live_code,
        num_out=num_out,
        batch_size=batch_size,
        resume=resume,
        output_dir=output_dir,
        final_file=output_dir / f"{model_short_name}_teacher_traces.json",
        batch_dir=output_dir / "batches",
        readable_file=output_dir / f"{model_short_name}_teacher_traces_readable.jsonl",
        pass_label="CoT",
    )

    if not no_cot:
        return

    # ── Pass 2: No-CoT ────────────────────────────────────────────────
    # Build [prompt][start_think][end_think][suffix] token sequences.
    # Mirrors attributions/probe_compression_exps.py:2041-2052.
    thinking = get_thinking_tokens(model_name)
    start_ids = thinking.get("start_token_ids")
    if start_ids is None:
        start_ids = tokenizer.encode(thinking["start_token"], add_special_tokens=False)
    end_ids = thinking.get("end_token_ids")
    if end_ids is None:
        end_ids = tokenizer.encode(thinking["end_token"], add_special_tokens=False)
    suffix_ids = tokenizer.encode(get_answer_suffix(data_name), add_special_tokens=False)

    print(
        f"\n[no-CoT] Prefix layout: start_ids={len(start_ids)} tok, "
        f"end_ids={len(end_ids)} tok, suffix_ids={len(suffix_ids)} tok"
    )

    nocot_prompts = []
    for prompt_text in cot_prompts:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        nocot_ids = prompt_ids + list(start_ids) + list(end_ids) + list(suffix_ids)
        nocot_prompts.append({"prompt_token_ids": nocot_ids})

    # No-CoT pass never emits reasoning — the huge CoT max_tokens budget is
    # wasteful, so cap at 1024 unless the user explicitly overrode it.
    nocot_gen_kwargs = dict(gen_kwargs)
    if nocot_gen_kwargs.get("max_tokens", 0) > 1024:
        nocot_gen_kwargs["max_tokens"] = 1024
    nocot_sampling_params = SamplingParams(
        n=num_out, logprobs=top_k_entropy,
        skip_special_tokens=False,
        **nocot_gen_kwargs,
    )

    _run_pass(
        llm=llm,
        sampling_params=nocot_sampling_params,
        prompts=nocot_prompts,
        reasoning_dataset=reasoning_dataset,
        tokenizer=tokenizer,
        data_name=data_name,
        model_name=model_name,
        code_meta=code_meta,
        is_live_code=is_live_code,
        num_out=num_out,
        batch_size=batch_size,
        resume=resume,
        output_dir=output_dir,
        final_file=output_dir / f"{model_short_name}_teacher_traces_nocot.json",
        batch_dir=output_dir / "batches_nocot",
        readable_file=output_dir / f"{model_short_name}_teacher_traces_nocot_readable.jsonl",
        pass_label="no-CoT",
    )


if __name__ == "__main__":
    fire.Fire(main)
