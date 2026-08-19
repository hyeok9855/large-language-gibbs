#! /bin/bash

# Motivating example (§4): sampling from simple distributions — full sweep for
# an arbitrary model.
#
# Distribution configurations:
#   Uniform:  {0,...,99} (paper), {25,...,74}, {-50,...,49}
#   Gaussian: (mean, std) = (0, 1) (paper), (-2.5, 1), (4.35, 1.09)
#   Mixture:  balanced and unbalanced bimodal Gaussians
#   Binomial: (n, p) = (20, 0.3)
#   Poisson:  rate = 4
#   Random walk (joint): d=8, X1 ~ N(0, var=1), Xi | X_{i-1} ~ N(X_{i-1}, var=0.5)
#   Multinomial (joint): d=8, N=100, equal probabilities (Gibbs B=2,4; B=1 is degenerate)
#
# Methods: indep, batch, direct, direct_continuation,
#          gibbs (B=1,4), gibbs_continuation (B=1,4);
#          instruct models additionally run barker and gambling gibbs.
#
# Usage:
#   bash sampling/run_parallel.sh <model_name> <port> [n_seeds=25] [reasoning=false] [temperature=1.0]
#
# model_name must be listed in MODEL_NAME_TO_TYPE (sampling/utils.py).
# reasoning=true enables manual reasoning (instruct models only).
# Requires the project venv to be active (e.g. via activate_here + export_here).
#
# Each distribution configuration is launched as one parallel wave
# (8-10 jobs x n_seeds); the script waits for a wave to finish before
# starting the next one. Existing result files are skipped by run.py.
# Visualise afterwards with: uv run python sampling/make_plot.py

# Directory containing this script, resolved regardless of the caller's CWD.
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

# Model type decides the method set (barker/gambling gibbs need an instruct model).
MODEL_TYPE=$(
    PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}" python -c "
import sys
from sampling.utils import MODEL_NAME_TO_TYPE

model = sys.argv[1]
if model not in MODEL_NAME_TO_TYPE:
    sys.exit(f'Unknown model_name: {model!r}. Add it to MODEL_NAME_TO_TYPE in sampling/utils.py.')
print(MODEL_NAME_TO_TYPE[model])
" "$MODEL"
) || exit 1

if [ "$REASONING" = true ] && [ "$MODEL_TYPE" != instruct ]; then
    echo "Error: manual reasoning is only supported for instruct models; got ${MODEL_TYPE} model '${MODEL}'." >&2
    exit 1
fi

if [ "$REASONING" = true ]; then
    REASONING_FLAG="--manual_reasoning"
else
    REASONING_FLAG=""
fi

echo "Model: $MODEL ($MODEL_TYPE), port: $PORT, seeds: $NSEEDS, reasoning: $REASONING, temperature: $TEMP"

mkdir -p "$SCRIPT_DIR/tmp"

# One pid file per invocation ($$ = this shell), so two sweeps sharing this
# checkout cannot truncate or delete each other's list.
PIDFILE="$SCRIPT_DIR/tmp/run_parallel.$$.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell, either of:
echo "To stop this sweep elsewhere: kill \$(cat $PIDFILE)"
echo "                          or: pkill -f \"run.py --model_name $MODEL\""
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/run.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

# Runs all methods and seeds for one distribution configuration, e.g.:
#   run_config --target uniform --minnum 0 --maxnum 99
run_config() {
    local COMMON_ARGS="--model_name $MODEL --port $PORT --temperature $TEMP $REASONING_FLAG $*"
    for SEED in $(seq 0 $(($NSEEDS - 1))); do
        # Independent, batch, and direct sampling
        launch $COMMON_ARGS --seed $SEED --methods indep
        launch $COMMON_ARGS --seed $SEED --methods batch
        launch $COMMON_ARGS --seed $SEED --methods direct --gibbs_k_vars 16
        launch $COMMON_ARGS --seed $SEED --methods direct_continuation --gibbs_k_vars 16

        # Gibbs sampling
        launch $COMMON_ARGS --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 1
        launch $COMMON_ARGS --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 4
        launch $COMMON_ARGS --seed $SEED --methods gibbs_continuation --gibbs_k_vars 16 --gibbs_block_size 1
        launch $COMMON_ARGS --seed $SEED --methods gibbs_continuation --gibbs_k_vars 16 --gibbs_block_size 4

        # Barker and Gambling gibbs variants (instruct models only)
        if [ "$MODEL_TYPE" = instruct ]; then
            launch $COMMON_ARGS --seed $SEED --methods barker_gibbs --gibbs_k_vars 16 --gibbs_block_size 1
            launch $COMMON_ARGS --seed $SEED --methods gambling_gibbs --gibbs_k_vars 16 --gibbs_block_size 1
        fi
    done
    wait
    PIDS=()
    : > "$PIDFILE"
}

# Uniform distributions
run_config --target uniform --minnum 0 --maxnum 99
run_config --target uniform --minnum 25 --maxnum 74
run_config --target uniform --minnum -50 --maxnum 49

# Gaussian distributions
run_config --target gaussian --mean 0.0 --std 1.0
run_config --target gaussian --mean -2.5 --std 1.0
run_config --target gaussian --mean 4.35 --std 1.09

# Multimodal: can the chain reach both modes, and in the right proportion?
run_config --target mixture --mixture_means -2.0 2.0 --mixture_stds 0.5 0.5 --mixture_weights 0.5 0.5
run_config --target mixture --mixture_means -2.0 2.0 --mixture_stds 0.5 0.5 --mixture_weights 0.8 0.2

# Shaped discrete supports, as a contrast to the flat uniform.
run_config --target binomial --n_trials 20 --p_success 0.3
run_config --target poisson --rate 4.0

# Joint distributions. n_samples is the number of vectors; indep/batch are skipped.
# BLOCKS: random walk allows B=1 (two-sided conditionals); multinomial needs B>=2,
# which is also why Barker/Gambling gibbs use BARKER_B instead of a fixed B=1.
run_joint_config() {
    local BLOCKS="$1"
    local BARKER_B="$2"
    shift 2
    local COMMON_ARGS="--model_name $MODEL --port $PORT --temperature $TEMP --gibbs_k_vars 8 $REASONING_FLAG $*"
    for SEED in $(seq 0 $(($NSEEDS - 1))); do
        launch $COMMON_ARGS --seed $SEED --methods direct --gibbs_k_vars 8
        launch $COMMON_ARGS --seed $SEED --methods direct_fixed --gibbs_k_vars 8
        launch $COMMON_ARGS --seed $SEED --methods direct_continuation --gibbs_k_vars 8
        launch $COMMON_ARGS --seed $SEED --methods direct_fixed_continuation --gibbs_k_vars 8
        for B in $BLOCKS; do
            launch $COMMON_ARGS --seed $SEED --methods gibbs --gibbs_k_vars 8 --gibbs_block_size $B
            launch $COMMON_ARGS --seed $SEED --methods gibbs_continuation --gibbs_k_vars 8 --gibbs_block_size $B
        done
        if [ "$MODEL_TYPE" = instruct ]; then
            launch $COMMON_ARGS --seed $SEED --methods barker_gibbs --gibbs_k_vars 8 --gibbs_block_size $BARKER_B
            launch $COMMON_ARGS --seed $SEED --methods gambling_gibbs --gibbs_k_vars 8 --gibbs_block_size $BARKER_B
        fi
    done
    wait
    PIDS=()
    : > "$PIDFILE"
}

run_joint_config "1 4" 1 --target random_walk --rw_x1_var 1.0 --rw_step_var 0.5
run_joint_config "2 4" 2 --target multinomial --multi_n 100

rm -f "$PIDFILE"

# remove tmp directory if it is empty
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null

echo "All jobs completed."
