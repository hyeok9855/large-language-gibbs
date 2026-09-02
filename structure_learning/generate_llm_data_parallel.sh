#! /bin/bash
# Usage: generate_llm_data_parallel.sh "<datasets>" "<sampling_methods>" <model_name> <port> [manual_reasoning=false] [nseeds=3]
#   datasets:         bnrep_tubercolosis bnrep_knowledge bnrep_algalactivity2 bnrep_gonorrhoeae bnrep_disputed1 bnrep_cardiovascular bnrep_consequenceCovid
#   sampling_methods: direct gibbs barker_gibbs gambling_gibbs
#
# Optional env overrides (unset = paper defaults):
#   TEMPERATURE    one temperature for every method
#   BLOCK_SIZE     one Gibbs block size for every dataset
#   NO_SWEEP=1     pass --no_sweep (random-block Gibbs)
#   SAVE_PROMPT=1  pass --save_prompt (writes <run>.prompt.md next to each CSV)
#   DRY_RUN=true   print the commands instead of running them
# generate_llm_data.py rescales thinning/burn_in from the block size, so never pin them here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

datasets=($1)
sampling_methods=($2)
model_name=$3
PORT=$4
manual_reasoning=${5:-false}
nseeds=${6:-3}

extra_opts=""
[ "$manual_reasoning" = true ] && extra_opts+=" --manual_reasoning"
[ -n "${NO_SWEEP:-}" ] && [ "$NO_SWEEP" != 0 ] && extra_opts+=" --no_sweep"
[ -n "${SAVE_PROMPT:-}" ] && [ "$SAVE_PROMPT" != 0 ] && extra_opts+=" --save_prompt"

# One pid file per invocation so concurrent sweeps do not clobber each other.
# Ctrl+C kills all launched jobs; from another shell: kill $(cat $PIDFILE)
mkdir -p "$SCRIPT_DIR/tmp"
PIDFILE="$SCRIPT_DIR/tmp/generate_llm_data.$$.pids"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "$PIDFILE"; exit 130' INT TERM

launch() {
    if [ "${DRY_RUN:-false}" = true ]; then
        echo "[dry-run] generate_llm_data.py $*"
        return
    fi
    python "$SCRIPT_DIR/generate_llm_data.py" "$@" &
    PIDS+=($!)
    echo $! >> "$PIDFILE"
}

default_block_size() {
    # Must match DATASET_PARAMS in train_dag_gflownet.py.
    case "$1" in
        bnrep_tubercolosis|bnrep_knowledge|bnrep_algalactivity2) echo 1 ;;
        bnrep_gonorrhoeae|bnrep_disputed1|bnrep_cardiovascular|bnrep_consequenceCovid) echo 2 ;;
        *) echo "Unknown dataset: $1 (no block_size configured)" >&2; exit 1 ;;
    esac
}

for dataset in "${datasets[@]}"; do
    if [ -n "${BLOCK_SIZE:-}" ]; then
        block_size=$BLOCK_SIZE
    else
        block_size=$(default_block_size "$dataset") || exit 1
    fi

    for sampling_method in "${sampling_methods[@]}"; do
        if [ -n "${TEMPERATURE:-}" ]; then
            temp=$TEMPERATURE
        elif [ "$sampling_method" = gambling_gibbs ] && [ "$manual_reasoning" != true ]; then
            temp=0.0  # gambling needs a deterministic bet decision unless it reasons first
        else
            temp=1.0
        fi

        for seed in $(seq 0 $((nseeds - 1))); do
            launch --dataset_name "$dataset" --port "$PORT" --model_name "$model_name" \
                --sampling_method "$sampling_method" --temperature "$temp" \
                --n_samples 200 --top_p 1.0 --n_chains 10 --block_size "$block_size" \
                --seed "$seed" $extra_opts
        done
    done
    wait  # one dataset at a time
    PIDS=()
    : > "$PIDFILE"
done

rm -f "$PIDFILE"
rmdir "$SCRIPT_DIR/tmp" 2>/dev/null
echo "All jobs completed."
