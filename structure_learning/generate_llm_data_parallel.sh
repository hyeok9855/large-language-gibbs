#! /bin/bash

# Directory containing this script, resolved regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/tmp"

PIDFILE="$SCRIPT_DIR/tmp/generate_llm_data.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell: kill $(cat structure_learning/tmp/generate_llm_data.pids)
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/generate_llm_data.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

datasets=($1)  # bnrep_tubercolosis bnrep_knowledge bnrep_algalactivity2 bnrep_disputed1 bnrep_consequenceCovid
sampling_methods=($2)  # direct gibbs barker_gibbs gambling_gibbs
model_name=$3  # e.g. meta-llama/Llama-3.1-8B, allenai/Olmo-3-32B-Think
PORT=$4
manual_reasoning=${5:-false}
nseeds=${6:-3}

# Whether manual reasoning actually applies is decided in generate_llm_data.py
# based on the model type (see MODEL_NAME_TO_TYPE).
if [ "$manual_reasoning" = true ]; then
    manual_reasoning_option="--manual_reasoning"
else
    manual_reasoning_option=""
fi

for dataset in ${datasets[@]}; do
    if [ "$dataset" == "bnrep_disputed1" ] || [ "$dataset" == "bnrep_consequenceCovid" ]; then
        block_size=2
    else
        block_size=1
    fi

    for sampling_method in ${sampling_methods[@]}; do
        if [ "$sampling_method" == "gambling_gibbs" ]; then
            temp=0.0
        else
            temp=1.0
        fi

        for seed in $(seq 0 $((nseeds - 1))); do
            ARGS="--port $PORT --model_name $model_name --sampling_method $sampling_method --temperature $temp --n_samples 200 --top_p 1.0 --n_chains 5 --block_size $block_size --seed $seed ${manual_reasoning_option}"
            launch --dataset_name $dataset $ARGS
        done
        wait
    done
done

rm -f "$PIDFILE"

# remove tmp directory if it is empty
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null

echo "All jobs completed."
