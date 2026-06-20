#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VLLM_PYTHON="${VLLM_PYTHON:-python}"
PROBE_PYTHON="${PROBE_PYTHON:-python}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
MODEL_SHORT="${MODEL##*/}"
TRAIN_DATA="${TRAIN_DATA:-MATH-500}"
DATASETS="${DATASETS:-MATH-500,opencompass/AIME2025,WildEval/ZebraLogic,fingertap/GPQA-Diamond}"
STAGES="${STAGES:-traces,attribution,probe,early_exit}"
TRACE_COUNT="${TRACE_COUNT:-16}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-16}"
PROBE_LAYER="${PROBE_LAYER:-23}"
CLUE_ALPHA="${CLUE_ALPHA:-0.5}"
PROBE_WINDOW="${PROBE_WINDOW:-256}"
EXIT_KS="${EXIT_KS:-1,2,5,10}"
FIXED_EXITS="${FIXED_EXITS:-50,70,80,90,95}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
OVERWRITE="${OVERWRITE:-False}"

contains_stage() {
    [[ ",${STAGES}," == *",$1,"* ]]
}

question_limit_args=()
if [[ -n "$MAX_QUESTIONS" ]]; then
    question_limit_args=(--max_questions "$MAX_QUESTIONS")
fi

IFS=',' read -r -a dataset_list <<< "$DATASETS"

if contains_stage traces; then
    for dataset in "${dataset_list[@]}"; do
        "$VLLM_PYTHON" scripts/generate_traces.py \
            --model_name "$MODEL" \
            --data_name "$dataset" \
            --num_out "$TRACE_COUNT" \
            --batch_size "$TRACE_BATCH_SIZE" \
            --no_cot False \
            "${question_limit_args[@]}"
    done
fi

if contains_stage attribution; then
    for dataset in "${dataset_list[@]}"; do
        "$VLLM_PYTHON" -m attributions.vllm_sentence_causal \
            --model "$MODEL" \
            --data_name "$dataset" \
            --semantic_guess_labels True \
            --clue_alpha "$CLUE_ALPHA" \
            --overwrite "$OVERWRITE" \
            "${question_limit_args[@]}"
    done
fi

train_short="${TRAIN_DATA##*/}"
attr_dir="outputs/${MODEL_SHORT}/${train_short}/contribution_graphs/sentence_causal/boxed"
probe_dir="${attr_dir}/solution_probe_three_way_last"

if contains_stage probe; then
    "$PROBE_PYTHON" -m attributions.train_solution_probe \
        --model "$MODEL" \
        --data_name "$TRAIN_DATA" \
        --attr_dir "$attr_dir" \
        --layer "$PROBE_LAYER" \
        --task_mode three_way \
        --sentence_aggregation last \
        --max_probe_input_tokens "$PROBE_WINDOW" \
        --epochs 10 \
        --patience 6 \
        --early_stop_metric macro_f1 \
        --probe_train_frac 0.5 \
        --probe_val_frac 0.1 \
        --probe_test_frac 0.1 \
        --output_dir "$probe_dir"
fi

if contains_stage early_exit; then
    for dataset in "${dataset_list[@]}"; do
        data_short="${dataset##*/}"
        subset="all"
        if [[ "$dataset" == "$TRAIN_DATA" ]]; then
            subset="eval"
        fi
        "$PROBE_PYTHON" -m attributions.solution_probe_early_exit_exps \
            --model "$MODEL" \
            --data_name "$dataset" \
            --probe_dir "$probe_dir" \
            --probe_layer "$PROBE_LAYER" \
            --attr_dir "outputs/${MODEL_SHORT}/${data_short}/contribution_graphs/sentence_causal/boxed" \
            --eval_question_subset "$subset" \
            --exit_ks "$EXIT_KS" \
            --fixed_exit_percentages "$FIXED_EXITS" \
            --max_probe_input_tokens "$PROBE_WINDOW" \
            --output_dir "${probe_dir}/early_exit_${data_short}" \
            "${question_limit_args[@]}"
    done
fi
