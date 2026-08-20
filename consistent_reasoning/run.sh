#!/usr/bin/env bash
# Reproduce Table 1 (consistent reasoning, Llama-3.1-8B base) from:
#   Choi et al., "Structured Inference with Large Language Gibbs" (2026)
#
# Prerequisites:
#   1. From repo root:  uv sync --extra consistent-reasoning
#   2. Start vLLM (separate terminal):
#        vllm serve meta-llama/Llama-3.1-8B --port 8000 --max-num-seqs 32
#      Or: source ../vllm_server.sh && start_vllm_server meta-llama/Llama-3.1-8B 8000 --max-num-seqs 32
#
# Usage:
#   cd /path/to/large-language-gibbs
#   bash consistent_reasoning/run.sh          # full Table 1
#   bash consistent_reasoning/run.sh smoke    # one quick Gibbs run
#   bash consistent_reasoning/run.sh gibbs        # Gibbs only (both testbeds)
#   bash consistent_reasoning/run.sh baselines  # zeroshot + npass only
#
# Every mode takes optional testbed names after it, so you can narrow the sweep:
#   bash consistent_reasoning/run.sh gibbs gsm8k                  # Gibbs, gsm8k only
#   CS_LIST=16 bash consistent_reasoning/run.sh gibbs gsm8k       # Gibbs cs=16, gsm8k only
#   CS_LIST="1 16" bash consistent_reasoning/run.sh gsm8k         # whole gsm8k sweep, cs in {1,16}
# Other list overrides: NPASS_LIST, ALPHA_LIST, TESTBEDS (default testbed set).
#
# Results: consistent_reasoning/eval_results/<testbed>/<algorithm>/<run_name>/summary.json
#          headline.mean_accuracy ± headline.std_accuracy  (3 partition seeds)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
PORT="${PORT:-8000}"
NPART="${NPART:-5}"              # paper Table 1: 3 seeds but here we use 5
SEED="${SEED:-42}"               # partition_base_seed
WORKERS="${WORKERS:-40}"
# ICM fans out one request per pool item per chunk, so num_workers chunks in
# flight means num_workers * chunk_size requests in flight; keep it modest or
# the vLLM queue pushes requests past their timeout into long retry backoffs.
WORKERS_ICM="${WORKERS_ICM:-$((WORKERS < 8 ? WORKERS : 8))}"
TEMP_GIBBS="${TEMP_GIBBS:-1.0}"
TEMP_NPASS="${TEMP_NPASS:-1.0}"
TEMP_ZEROSHOT="${TEMP_ZEROSHOT:-0.0}"   # paper: greedy zero-shot
CS_LIST="${CS_LIST:-1 4 16}"     # chunk_size_cis values swept by gibbs / direct-npass
# GIBBS_ORDER_FLAG=--fast_demo_order draws the demo order once per sweep so vLLM's
# prefix cache hits (what makes cs=16 tractable). It approximates the default sampler,
# so those runs land under a separate *_fastorder name.
GIBBS_ORDER_FLAG="${GIBBS_ORDER_FLAG:-}"
NPASS_LIST="${NPASS_LIST:-1 4}"  # per-group npass baselines
ALPHA_LIST="${ALPHA_LIST:-1.0 3.0 10.0 30.0}"   # ICM alphas
TESTBEDS_DEFAULT="${TESTBEDS:-truthfulQA gsm8k}"

run_eval() {
  echo ""
  echo "========== $* =========="
  uv run python consistent_reasoning/run_eval.py "$@"
}

run_gibbs_testbed() {
  local testbed="$1"
  for cs in $CS_LIST; do
    run_eval \
      --testbed "$testbed" \
      --algorithm gibbs \
      --model "$MODEL" \
      --port "$PORT" \
      --temperature "$TEMP_GIBBS" \
      --chunk_size_cis "$cs" \
      $GIBBS_ORDER_FLAG \
      --n_partitions "$NPART" \
      --partition_base_seed "$SEED" \
      --num_workers "$WORKERS"
  done
}

run_baselines_testbed() {
  local testbed="$1"

  run_eval \
    --testbed "$testbed" \
    --algorithm zeroshot \
    --model "$MODEL" \
    --port "$PORT" \
    --temperature "$TEMP_ZEROSHOT" \
    --n_partitions 1 \
    --num_workers "$WORKERS"

  # Per-group npass (Table 1): each consistency group labeled on its own.
  for np in $NPASS_LIST; do
    run_eval \
      --testbed "$testbed" \
      --algorithm npass \
      --model "$MODEL" \
      --port "$PORT" \
      --temperature "$TEMP_NPASS" \
      --n_passes "$np" \
      --chunk_size_cis 1 \
      --n_partitions "$NPART" \
      --partition_base_seed "$SEED" \
      --num_workers "$WORKERS"
  done

  # "Direct" baselines matched to the Gibbs runs: 25 autoregressive passes over
  # the chunk, majority vote (Gibbs likewise majority-votes 25 thinned samples).
  #   - shuffled (default): each pass uses a fresh block-preserving random order,
  #     i.e. ancestral sampling marginalized over orders.
  #   - fixed_order: every pass uses the same (original) order, i.e. pure
  #     autoregressive generation under one order -> exposes order-dependent bias.
  for cs in $CS_LIST; do
    for order_flag in "" "--fixed_order"; do
      run_eval \
        --testbed "$testbed" \
        --algorithm npass \
        --model "$MODEL" \
        --port "$PORT" \
        --temperature "$TEMP_NPASS" \
        --n_passes 25 \
        --chunk_size_cis "$cs" \
        $order_flag \
        --n_partitions "$NPART" \
        --partition_base_seed "$SEED" \
        --num_workers "$WORKERS"
    done
  done
}

run_icm_testbed() {
  local testbed="$1"
  for alpha in $ALPHA_LIST; do
    run_eval \
      --testbed "$testbed" \
      --algorithm icm \
      --model "$MODEL" \
      --port "$PORT" \
      --chunk_size_cis 16 \
      --alpha "$alpha" \
      --n_partitions "$NPART" \
      --partition_base_seed "$SEED" \
      --num_workers "$WORKERS_ICM"
  done
}

run_testbed() {
  local testbed="$1"
  run_baselines_testbed "$testbed"
  run_gibbs_testbed "$testbed"
  run_icm_testbed "$testbed"
}

run_smoke() {
  run_eval \
    --testbed truthfulQA \
    --algorithm gibbs \
    --model "$MODEL" \
    --port "$PORT" \
    --temperature "$TEMP_GIBBS" \
    --chunk_size_cis 1 \
    --n_partitions 1 \
    --only_partition 0 \
    --num_workers 4
}

run_full() {
  for testbed in "$@"; do
    run_testbed "$testbed"
  done
  echo ""
  echo "Done. Summaries under: $REPO_ROOT/consistent_reasoning/eval_results/"
}

MODE="${1:-full}"
shift || true
# Remaining args (if any) select the testbeds; otherwise sweep the defaults.
TESTBEDS_SEL=("$@")
if [ "${#TESTBEDS_SEL[@]}" -eq 0 ]; then
  read -r -a TESTBEDS_SEL <<< "$TESTBEDS_DEFAULT"
fi

echo "Model=$MODEL  port=$PORT  n_partitions=$NPART  seed=$SEED  workers=$WORKERS (icm: $WORKERS_ICM)"
echo "mode=$MODE  testbeds=${TESTBEDS_SEL[*]}  chunk_sizes=$CS_LIST"

case "$MODE" in
  smoke)     run_smoke ;;
  baselines) for t in "${TESTBEDS_SEL[@]}"; do run_baselines_testbed "$t"; done ;;
  icm)       for t in "${TESTBEDS_SEL[@]}"; do run_icm_testbed "$t"; done ;;
  gibbs)     for t in "${TESTBEDS_SEL[@]}"; do run_gibbs_testbed "$t"; done ;;
  full)      run_full "${TESTBEDS_SEL[@]}" ;;
  truthfulQA|gsm8k) run_full "$MODE" ;;   # mode name doubles as a testbed selector
  *)         echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
