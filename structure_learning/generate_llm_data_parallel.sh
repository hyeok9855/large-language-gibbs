#! /bin/bash

# Directory containing this script, resolved regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/tmp"

PIDFILE="$SCRIPT_DIR/tmp/generate_llm_data.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell: kill $(cat structure_learning/scripts/tmp/generate_llm_data.pids)
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/generate_llm_data.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

datasets=($1)  # bnrep_tubercolosis bnrep_knowledge bnrep_algalactivity2 bnrep_disputed1 bnrep_consequenceCovid
sampling_methods=($2)  # direct gibbs barker_gibbs gambling_gibbs
PORT=$3
llm_name=${4}  # Llama8B, Olmo32B, Llama70B
base_or_instruct=${5}  # base, instruct
manual_reasoning=${6:-false}
nseeds=${7:-3}

if [ "$base_or_instruct" == "instruct" ] && [ "$manual_reasoning" = true ]; then
    manual_reasoning_option="--manual_reasoning"
else
    manual_reasoning_option=""
fi

if [ "$base_or_instruct" == "instruct" ]; then
    if [ "$llm_name" == "Llama70B" ]; then
        llm_id="meta-llama/Llama-3.1-70B-Instruct"
    elif [ "$llm_name" == "Llama8B" ]; then
        llm_id="meta-llama/Llama-3.1-8B-Instruct"
    elif [ "$llm_name" == "Olmo32B" ]; then
        llm_id="allenai/Olmo-3-32B-Think"
    elif [ "$llm_name" == "Gemma31B" ]; then
        llm_id="google/gemma-4-31B-it"
    fi
else
    if [ "$llm_name" == "Llama70B" ]; then
        llm_id="meta-llama/Llama-3.1-70B"
    elif [ "$llm_name" == "Llama8B" ]; then
        llm_id="meta-llama/Llama-3.1-8B"
    elif [ "$llm_name" == "Olmo32B" ]; then
        llm_id="allenai/Olmo-3-1125-32B"
    elif [ "$llm_name" == "Gemma31B" ]; then
        llm_id="google/gemma-4-31B"
    fi
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

        for seed in $(seq 1 $nseeds); do
            ARGS="--base_url http://localhost:$PORT/v1 --model_name $llm_id --sampling_method $sampling_method --temperature $temp --n_samples 200 --top_p 1.0 --n_chains 5 --block_size $block_size --seed $seed ${manual_reasoning_option}"
            launch --dataset_name $dataset $ARGS
        done
        wait
    done
done

rm -f "$PIDFILE"

# remove tmp directory if it is empty
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null

echo "All jobs completed."

# python generate_llm_data.py --dataset_name bnrep_tubercolosis --base_url http://localhost:8000/v1 --model_name meta-llama/Llama-3.1-8B --sampling_method direct --temperature 1.0 --n_samples 200 --top_p 1.0 --n_chains 5 --seed 1
