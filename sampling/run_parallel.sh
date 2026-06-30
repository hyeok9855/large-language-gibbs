#! /bin/bash

# Directory containing this script, resolved regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/tmp"

PIDFILE="$SCRIPT_DIR/tmp/run_multiseed.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell: kill $(cat sampling/tmp/run_multiseed.pids)
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/run.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

TARGET=$1  # uniform (discrete), gaussian (continuous)
MODEL=$2  # meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct, allenai/Olmo-3-1125-32B, allenai/Olmo-3-32B-Think
PORT=$3
TEMP=${4:-1.0}
NSEEDS=${5:-25}
REASONING=${6:-false}

if [ "$REASONING" = true ]; then
    REASONING_FLAG="--manual_reasoning"
else
    REASONING_FLAG=""
fi

COMMON_ARGS="--target $TARGET --model_name $MODEL --port $PORT --temperature $TEMP $REASONING_FLAG"
for SEED in $(seq 0 $(($NSEEDS - 1))); do
    # Independent, batch, and direct sampling
    launch $COMMON_ARGS --seed $SEED --methods indep
    launch $COMMON_ARGS --seed $SEED --methods batch
    launch $COMMON_ARGS --seed $SEED --methods direct --gibbs_k_vars 16

    # Gibbs sampling
    launch $COMMON_ARGS --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 1
    launch $COMMON_ARGS --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 4

    # Barker and Gambling variants
    launch $COMMON_ARGS --seed $SEED --methods barker --gibbs_k_vars 16 --gibbs_block_size 1
    launch $COMMON_ARGS --seed $SEED --methods gambling --gibbs_k_vars 16 --gibbs_block_size 1
done
wait
rm -f "$PIDFILE"

# remove tmp directory if it is empty
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null

echo "All jobs completed."
