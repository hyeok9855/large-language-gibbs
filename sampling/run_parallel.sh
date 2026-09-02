#! /bin/bash

# JSON-family sweep over the simple-distribution targets.
# The continuation family has its own sweep: sampling_continuation/run_parallel.sh.
#
# Usage:
#   bash sampling/run_parallel.sh <model_name> <port> [n_seeds=25] [reasoning=false] [temperature=1.0]
#
# model_name must be listed in MODEL_NAME_TO_TYPE (common/utils.py); the
# project venv must be active. Each config runs as one parallel wave of
# ~7 jobs x n_seeds; existing result files are skipped by run.py.
# Visualise afterwards with: uv run python sampling/make_plot.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=$1
PORT=$2
NSEEDS=${3:-25}
REASONING=${4:-false}
TEMP=${5:-1.0}

if [ -z "$MODEL" ] || [ -z "$PORT" ]; then
    echo "Usage: bash sampling/run_parallel.sh <model_name> <port> [n_seeds=25] [reasoning=false] [temperature=1.0]"
    exit 1
fi

# Base vs instruct decides whether the barker/gambling kernels run at all.
MODEL_TYPE=$(
    PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}" python -c "
import sys
from common.utils import MODEL_NAME_TO_TYPE

model = sys.argv[1]
if model not in MODEL_NAME_TO_TYPE:
    sys.exit(f'Unknown model_name: {model!r}. Add it to MODEL_NAME_TO_TYPE in common/utils.py.')
print(MODEL_NAME_TO_TYPE[model])
" "$MODEL"
) || exit 1

if [ "$REASONING" = true ]; then
    if [ "$MODEL_TYPE" != instruct ]; then
        echo "Error: reasoning=true requires an instruct model; '${MODEL}' is a ${MODEL_TYPE} model." >&2
        exit 1
    fi
    REASONING_FLAG="--manual_reasoning"
else
    REASONING_FLAG=""
fi

# Reasoning traces for seed 0 only; see sampling/reasoning_traces.py.
trace_flag() {  # trace_flag <seed>
    if [ "$REASONING" = true ] && [ "$1" -eq 0 ]; then
        echo "--n_traces 10"
    else
        echo "--n_traces 0"
    fi
}

echo "Model: $MODEL ($MODEL_TYPE), port: $PORT, seeds: $NSEEDS, reasoning: $REASONING, temperature: $TEMP"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/run.py" "$@" &
    PIDS+=($!)
}

# Manual reasoning is defined only for the decision kernels (REASONING_METHODS
# in sampling/run.py), so the field methods are skipped under reasoning=true.
launch_field() {
    [ "$REASONING" = true ] && return 0
    launch "$@"
}

# Runs all methods and seeds for one distribution configuration, e.g.:
#   run_config --target uniform --minnum 0 --maxnum 99
run_config() {
    local COMMON_ARGS="--model_name $MODEL --port $PORT --temperature $TEMP $REASONING_FLAG $*"
    for SEED in $(seq 0 $(($NSEEDS - 1))); do
        SEED_ARGS="--seed $SEED $(trace_flag $SEED)"

        launch_field $COMMON_ARGS $SEED_ARGS --methods indep
        launch_field $COMMON_ARGS $SEED_ARGS --methods batch

        launch_field $COMMON_ARGS $SEED_ARGS --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 1
        launch_field $COMMON_ARGS $SEED_ARGS --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 4

        if [ "$MODEL_TYPE" = instruct ]; then
            launch $COMMON_ARGS $SEED_ARGS --methods barker_gibbs --gibbs_k_vars 16 --gibbs_block_size 1
            launch $COMMON_ARGS $SEED_ARGS --methods gambling_gibbs --gibbs_k_vars 16 --gibbs_block_size 1
        fi
    done
    wait
    PIDS=()
}

run_config --target uniform --minnum 0 --maxnum 99
run_config --target gaussian --mean 0.0 --std 1.0
run_config --target uniform --minnum -50 --maxnum 49
run_config --target gaussian --mean -2.5 --std 1.0
run_config --target mixture --mixture_means -2.0 2.0 --mixture_stds 0.5 0.5 --mixture_weights 0.8 0.2
run_config --target binomial --n_trials 20 --p_success 0.3
run_config --target poisson --rate 4.0
