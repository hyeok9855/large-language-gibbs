#!/bin/bash

TARGET=$1  # uniform (discrete), gaussian (continuous)
MODEL=$2  # meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct, allenai/Olmo-3-1125-32B, allenai/Olmo-3-32B-Think
PORT=$3
NSEEDS=${4:-25}
REASONING=${5:-false}

mkdir -p "$(dirname "$0")/tmp"

PIDFILE="$(dirname "$0")/tmp/run_multiseed.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell: kill $(cat sampling/tmp/run_multiseed.pids)
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

if [ "$REASONING" = true ]; then
    REASONING_FLAG="--manual_reasoning"
else
    REASONING_FLAG=""
fi

COMMON_ARGS="--target $TARGET --model_name $MODEL --port $PORT $REASONING_FLAG"

launch() {
    python run.py "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

for SEED in $(seq 0 $(($NSEEDS - 1))); do
    # Independent and batch sampling
    launch $COMMON_ARGS --seed $SEED --methods indep batch

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
rmdir "$(dirname "$0")/tmp" 2>/dev/null

echo "All jobs completed."
