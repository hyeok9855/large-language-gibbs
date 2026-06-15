#!/bin/bash

TARGET=$1  # uniform (discrete), gaussian (continuous)
MODEL=$2  # meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct, allenai/Olmo-3-1125-32B, allenai/Olmo-3-32B-Think
PORT=$3
NSEEDS=${4:-25}

for SEED in $(seq 0 $(($NSEEDS - 1))); do
    # Independent and batch sampling
    python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods indep batch &

    # Gibbs sampling
    python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 1 &
    python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods gibbs --gibbs_k_vars 16 --gibbs_block_size 4 &

    # Barker and Gambling variants
    python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods barker --gibbs_k_vars 16 --gibbs_block_size 1 &
    python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods gambling --gibbs_k_vars 16 --gibbs_block_size 1 &

    # Barker and Gambling variants with manual reasoning
    # python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods barker --gibbs_k_vars 16 --gibbs_block_size 1 --manual_reasoning &
    # python run.py --target $TARGET --model-name $MODEL --port $PORT --seed $SEED --methods gambling --gibbs_k_vars 16 --gibbs_block_size 1 --manual_reasoning &
done
wait
echo "All jobs completed."
