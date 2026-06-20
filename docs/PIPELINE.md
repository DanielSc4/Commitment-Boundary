# Practical pipeline

This document translates the paper’s method into the concrete operations
performed by the code.

## 1. Sample complete reasoning traces

`scripts/generate_traces.py` loads a benchmark, applies the model’s chat
template, and samples complete reasoning traces with vLLM. It saves token IDs,
not only decoded text, because model-specific beginning/end-of-thinking markers
must survive exactly.

Default generation:

- temperature: `0.7`
- top-p: `0.9`
- maximum new tokens: `16,384`
- GPT-OSS: 16 traces per question in the paper
- Gemma/Qwen: 8 traces per question in the paper

Output:

```text
data/<dataset-short-name>/<model-short-name>_teacher_traces.json
```

Each question stores the prompt token IDs, sampled trace token IDs, decoded
traces, extracted answers, ground truth, and token entropies.

## 2. Measure answer formation by truncation

`attributions/vllm_sentence_causal.py` is the central causal measurement.

For a reasoning trace with sentences `C1 ... Cn`, it constructs:

```text
X0 = prompt + BOT + EOT + answer suffix
X1 = prompt + BOT + C1 + EOT + answer suffix
...
Xn = prompt + BOT + C1 ... Cn + EOT + answer suffix
```

The answer suffix is `Therefore, the final answer is \boxed{` for the paper’s
math and multiple-choice tasks. Prefixes are evaluated in batches with vLLM.

For every prefix, the script records:

- the greedily elicited answer;
- the probability of the full-CoT answer’s first token;
- semantic equivalence to the full-CoT answer;
- the sentence-level probability change from the previous prefix.

Math equivalence uses `math-verify`; multiple-choice answers are normalized to
their option letter. The comparison target is the model’s own full-CoT answer,
not the benchmark ground truth. This isolates *answer stabilization* from
whether the answer happens to be correct.

### Trace filtering

The implementation excludes traces that cannot identify a meaningful
CoT-induced transition, including:

- missing end-of-thinking markers;
- empty sentence segmentation;
- identical full-CoT and no-CoT elicited responses;
- insufficient full-vs-no-CoT first-token probability gain;
- generation or memory failures.

Semantic first-token collisions are recorded in the attribution metadata so
they can be audited. These filters mean the analysis is conditional on traces
where explicit reasoning materially improves answer formation.

### Three answer-formation labels

Let `p0` be no-CoT confidence and `pn` full-CoT confidence. The confidence gate
is:

```text
tau = p0 + clue_alpha * (pn - p0)
```

The paper uses `clue_alpha=0.5`.

- `0 / no_guess`: confidence does not cross `tau`;
- `1 / mid_guess`: confidence crosses `tau`, but the answer differs
  semantically from the full-CoT answer;
- `2 / final_guess`: confidence crosses `tau` and the answer is equivalent to
  the full-CoT answer.

Attribution files are written to:

```text
outputs/<model>/<dataset>/contribution_graphs/sentence_causal/boxed/question_XXXX.json
```

## 3. Locate the commitment boundary

For the full-answer confidence sequence, the boundary is the sentence with the
largest positive confidence jump:

```text
i* = argmax_i (p_i - p_{i-1})
```

The prefix ending at `i*` is tested as a sufficient decision point. Sentences
after it are the candidate epiphenomenal tail: text the model continues to
generate after its answer has stabilized.

## 4. Extract activations and train the probe

`attributions/train_solution_probe.py` reads the attribution labels and
collects residual-stream states at a chosen transformer layer. The default
`last` aggregation keeps the final token state of each sentence.

The split is by question, never by sentence:

- 50% probe train;
- 10% validation;
- 10% probe test;
- 30% held out for the paper’s early-exit evaluation.

The three-way probe receives only the current and previous sentence states. It
uses layer normalization, a learned causal attention pool, a learned local
projection of the current sentence, ReLU, and a three-class output layer. A
sliding lookback window prevents future leakage.

Paper training defaults:

- Adam, learning rate `1e-4`;
- weight decay `1e-4`;
- batch size `64`;
- dropout `0.1`;
- gradient clipping `1.0`;
- class-weighted cross entropy, weights capped at `5`;
- 10 epochs, early stopping patience `6`;
- checkpoint selected by validation macro-F1.

Activation caches and checkpoints are generated artifacts under `outputs/`.

For layer/window sweeps, first cache several layers in one model pass:

```bash
python -m attributions.precollect_solution_probe_cache \
  --model openai/gpt-oss-20b \
  --data_name MATH-500 \
  --layers 19,22,23
```

Then train configurations with `--skip_collection True` and summarize them:

```bash
python -m attributions.summarize_solution_probe_sweep \
  --sweep_dir <sweep-directory>
```

## 5. Probe-guided early exit

`attributions/solution_probe_early_exit_exps.py` scores each sentence in causal
order. An exit is triggered after `k` consecutive `final_guess` predictions.
Requiring several consecutive predictions is the safety/accuracy control used
for the paper’s trade-off curves.

At the chosen sentence:

1. discard the remaining CoT tokens;
2. append the model’s end-of-thinking marker;
3. append the task answer suffix;
4. greedily decode the answer.

If no trigger occurs, evaluation falls back to the full CoT.

Compared conditions:

- full CoT;
- no CoT;
- fixed-percentage sentence truncation;
- probe exit for `k = 1, 2, 5, 10`;
- optional class-2 probability or margin thresholds.

The output summary reports accuracy, token savings, fallback rate, boundary
detection, early-fire rate, delay, and Pareto-efficient operating points.

## 6. Numeric perturbation stress test

`attributions/vllm_cot_number_perturbation.py` consumes attribution files and
randomly offsets numeric literals by values from `{-5,...,-1,1,...,5}`.

The paper compares:

- `pre_boundary_early_exit`: keep only the prefix through `i*`, then corrupt
  numbers in that prefix;
- `post_boundary`: keep the full CoT, but corrupt only numbers after `i*`.

If post-boundary text is causally redundant, its corruption should preserve the
original full-CoT answer more often than corruption before the boundary.

## Reproducing another model family

Change `MODEL`, `PROBE_LAYER`, and `TRACE_COUNT`. The code knows the thinking
markers and decoder layouts for GPT-OSS, Qwen3, and Gemma 4. Run a layer/window
sweep before choosing the final probe; selected layers are model-specific.
