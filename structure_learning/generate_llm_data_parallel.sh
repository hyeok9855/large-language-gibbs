#! /bin/bash

# Directory containing this script, resolved regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/tmp"

# One pid file per invocation ($$ = this shell), so two sweeps sharing this
# checkout cannot truncate or delete each other's list.
PIDFILE="$SCRIPT_DIR/tmp/generate_llm_data.$$.pids"
PIDS=()
: > "$PIDFILE"

# Ctrl+C kills all launched jobs. From another shell: kill $(cat $PIDFILE)
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    python "$SCRIPT_DIR/generate_llm_data.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

datasets=($1)  # bnrep_tubercolosis bnrep_knowledge bnrep_algalactivity2 bnrep_gonorrhoeae bnrep_disputed1 bnrep_cardiovascular bnrep_consequenceCovid
sampling_methods=($2)  # direct gibbs direct_continuation gibbs_continuation barker_gibbs gambling_gibbs
model_name=$3  # e.g. meta-llama/Llama-3.1-8B, allenai/Olmo-3.1-32B-Instruct
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
    # Must match DATASET_PARAMS in train_dag_gflownet.py.
    case "$dataset" in
        bnrep_tubercolosis|bnrep_knowledge|bnrep_algalactivity2)
            block_size=1 ;;
        bnrep_gonorrhoeae|bnrep_disputed1|bnrep_cardiovascular|bnrep_consequenceCovid)
            block_size=2 ;;
        *)
            echo "Unknown dataset: $dataset (no block_size configured)" >&2
            exit 1 ;;
    esac

    for sampling_method in ${sampling_methods[@]}; do
        if [ "$sampling_method" == "gambling_gibbs" ]; then
            temp=0.0
        else
            temp=1.0
        fi

        for seed in $(seq 0 $((nseeds - 1))); do
            ARGS="--port $PORT --model_name $model_name --sampling_method $sampling_method --temperature $temp --n_samples 200 --top_p 1.0 --n_chains 10 --block_size $block_size --seed $seed ${manual_reasoning_option}"
            launch --dataset_name $dataset $ARGS
        done
    done
    wait
    PIDS=()
    : > "$PIDFILE"
done

rm -f "$PIDFILE"

# remove tmp directory if it is empty
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null

echo "All jobs completed."
