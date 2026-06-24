"""Evaluate an unsupervised-elicitation algorithm against a fixed eval set
with random partitioning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np

from consistent_reasoning.models import (
    ModelAPI,
    OpenAICompatLLM,
    ReasoningOpenAICompatLLM,
    setup_environment,
)
from consistent_reasoning.dataloaders import (
    load_eval_set,
    build_partitions,
    build_cid_index,
    select_items_for_chunk,
    load_processed_train,
    initialize,
    CONSISTENT_REASONING_DIR,
)
from consistent_reasoning.algorithms.icm import run_icm_search
from consistent_reasoning.algorithms.gibbs import run_gibbs_search
from consistent_reasoning.algorithms.barker_gibbs import run_barker_gibbs_search
from consistent_reasoning.algorithms.gambling_gibbs import run_gambling_gibbs_search
from consistent_reasoning.algorithms.npass import run_npass_search
from consistent_reasoning.algorithms.zeroshot import run_zeroshot_search

_print_lock = threading.Lock()


def _chunk_log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


# --- Hyperparam handling ----------------------------------------------------

INSTRUCT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "allenai/Olmo-3-32B-Think",
]

ICM_CACHE_KEY_FIELDS = (
    "K",
    "alpha",
    "num_seed",
    "decay",
    "initial_T",
    "final_T",
    "scheduler",
    "no_trailing_space",
    "model",
    "instruction_tuned",
    "system_prompt",
)
GIBBS_CACHE_KEY_FIELDS = (
    "temperature",
    "burn_in",
    "thinning",
    "num_samples",
    "sweep",
    "no_trailing_space",
    "model",
    "instruction_tuned",
    "system_prompt",
)
ZEROSHOT_CACHE_KEY_FIELDS = (
    "temperature",
    "no_trailing_space",
    "model",
    "instruction_tuned",
    "system_prompt",
)
NPASS_CACHE_KEY_FIELDS = (
    "temperature",
    "n_passes",
    "all_passes",
    "no_trailing_space",
    "model",
    "instruction_tuned",
    "system_prompt",
)
BARKER_GIBBS_CACHE_KEY_FIELDS = GIBBS_CACHE_KEY_FIELDS
GAMBLING_GIBBS_CACHE_KEY_FIELDS = (*GIBBS_CACHE_KEY_FIELDS, "manual_reasoning")
_CACHE_KEY_FIELDS_BY_ALGO = {
    "icm": ICM_CACHE_KEY_FIELDS,
    "gibbs": GIBBS_CACHE_KEY_FIELDS,
    "zeroshot": ZEROSHOT_CACHE_KEY_FIELDS,
    "npass": NPASS_CACHE_KEY_FIELDS,
    "barker_gibbs": BARKER_GIBBS_CACHE_KEY_FIELDS,
    "gambling_gibbs": GAMBLING_GIBBS_CACHE_KEY_FIELDS,
}


def relevant_args_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    try:
        fields = _CACHE_KEY_FIELDS_BY_ALGO[args.algorithm]
    except KeyError as e:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}") from e
    snapshot = {f: getattr(args, f) for f in fields}
    if args.algorithm not in ("zeroshot", "npass"):
        snapshot["chunk_size_cis"] = args.chunk_size_cis
    snapshot["algorithm"] = args.algorithm
    return snapshot


# --- Chunk runners ----------------------------------------------------------


def _chunk_cache_path(output_dir: Path, partition_index: int, chunk_index: int) -> Path:
    return output_dir / f"partition_{partition_index:02d}" / f"chunk_{chunk_index:02d}.json"


def _read_cached_chunk(
    cache_path: Path,
    *,
    expected_chunk_cids: list[int],
    expected_partition_seed: int,
    expected_args_snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open() as f:
            cached = json.load(f)
    except json.JSONDecodeError:
        return None

    if sorted(cached.get("chunk_consistency_ids", [])) != sorted(expected_chunk_cids):
        return None
    if cached.get("partition_seed") != expected_partition_seed:
        return None
    if cached.get("cache_key_args") != expected_args_snapshot:
        return None
    return cached


def run_icm_chunk(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    partition_index: int,
    partition_seed: int,
    chunk_index: int,
    output_dir: Path,
    eval_set_meta: dict[str, Any],
    model_api: Any,
    run_name: str,
) -> dict[str, Any]:
    chunk_cids = [item["consistency_id"] for item in items]
    cache_args = relevant_args_snapshot(args)
    cache_path = _chunk_cache_path(output_dir, partition_index, chunk_index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _read_cached_chunk(
        cache_path,
        expected_chunk_cids=chunk_cids,
        expected_partition_seed=partition_seed,
        expected_args_snapshot=cache_args,
    )
    if cached is not None:
        _chunk_log(f"[p{partition_index:02d} c{chunk_index:02d}] cache hit, skipping")
        return cached

    random.seed(partition_seed)
    np.random.seed(partition_seed)

    fewshot_ids = list(range(len(items)))
    demonstrations, _, whole_ids, _ = initialize(items, fewshot_ids, args)

    pipeline_name = (
        f"runeval_{run_name}_p{partition_index:02d}_c{chunk_index:02d}_seed{partition_seed}"
    )
    log_path = cache_path.with_suffix(".log.jsonl")
    log_path.unlink(missing_ok=True)

    t0 = time.time()
    final_demos, final_metric = run_icm_search(
        demonstrations,
        whole_ids,
        args,
        model_api,
        pipeline_name=pipeline_name,
        log_path=log_path,
    )
    duration = time.time() - t0

    predictions = []
    for uid in sorted(final_demos.keys()):
        item = final_demos[uid]
        predictions.append(
            {
                "uid_in_chunk": int(uid),
                "source_index": int(item["_source_index"]),
                "consistency_id": item["consistency_id"],
                "vanilla_label": int(bool(item["vanilla_label"])),
                "predicted_label": (int(item["label"]) if item.get("label") is not None else None),
                "init_type": item.get("type"),
            }
        )

    n_predicted = sum(1 for p in predictions if p["predicted_label"] is not None)
    n_correct = sum(
        1
        for p in predictions
        if p["predicted_label"] is not None and p["predicted_label"] == p["vanilla_label"]
    )
    record = {
        "algorithm": args.algorithm,
        "testbed": args.testbed,
        "run_name": run_name,
        "partition_index": int(partition_index),
        "chunk_index": int(chunk_index),
        "partition_seed": int(partition_seed),
        "chunk_consistency_ids": chunk_cids,
        "n_items": len(items),
        "n_predicted": int(n_predicted),
        "duration_seconds": float(duration),
        "cache_key_args": cache_args,
        "eval_set_meta": {
            k: eval_set_meta[k]
            for k in ("testbed", "source_file", "n_consistency_ids", "seed")
            if k in eval_set_meta
        },
        "final_metric_summary": {
            "train_accuracy": float(final_metric["train_accuracy"]),
            "train_size": int(final_metric["train_size"]),
            "predict_distribution": dict(final_metric["train_predict_distribution"]),
            "label_distribution": dict(final_metric["train_label_distribution"]),
        },
        "chunk_accuracy": float(n_correct / n_predicted) if n_predicted else None,
        "predictions": predictions,
    }
    with cache_path.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    _chunk_log(
        f"[p{partition_index:02d} c{chunk_index:02d}] done in {duration:.1f}s, "
        f"acc={record['chunk_accuracy']}"
    )
    return record


def _run_label_predict_chunk(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    partition_index: int,
    partition_seed: int,
    chunk_index: int,
    output_dir: Path,
    eval_set_meta: dict[str, Any],
    model_api: Any,
    run_name: str,
    search_fn: Any,
    verbose_steps: bool = True,
) -> dict[str, Any]:
    chunk_cids = [item["consistency_id"] for item in items]
    cache_args = relevant_args_snapshot(args)
    cache_path = _chunk_cache_path(output_dir, partition_index, chunk_index)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _read_cached_chunk(
        cache_path,
        expected_chunk_cids=chunk_cids,
        expected_partition_seed=partition_seed,
        expected_args_snapshot=cache_args,
    )
    if cached is not None:
        _chunk_log(f"[p{partition_index:02d} c{chunk_index:02d}] cache hit, skipping")
        return cached

    random.seed(partition_seed)
    np.random.seed(partition_seed)

    demonstrations: dict[int, dict[str, Any]] = {}
    whole_ids: list[int] = []
    for uid, item in enumerate(items):
        item = deepcopy(item)
        item["vanilla_label"] = item["label"]
        item["uid"] = uid
        item["label"] = None
        item["type"] = "predict"
        demonstrations[uid] = item
        whole_ids.append(uid)

    log_path = cache_path.with_suffix(".log.jsonl")
    log_path.unlink(missing_ok=True)

    t0 = time.time()
    final_demos, final_metric = search_fn(
        demonstrations,
        whole_ids,
        args,
        model_api,
        log_path=log_path,
        verbose_steps=verbose_steps,
    )
    duration = time.time() - t0

    predictions = []
    for uid in sorted(final_demos.keys()):
        item = final_demos[uid]
        predictions.append(
            {
                "uid_in_chunk": int(uid),
                "source_index": int(item["_source_index"]),
                "consistency_id": item["consistency_id"],
                "vanilla_label": int(bool(item["vanilla_label"])),
                "predicted_label": (int(item["label"]) if item.get("label") is not None else None),
                "predicted_score": item.get("_predicted_score"),
                "init_type": item.get("type"),
            }
        )

    n_predicted = sum(1 for p in predictions if p["predicted_label"] is not None)
    n_correct = sum(
        1
        for p in predictions
        if p["predicted_label"] is not None and p["predicted_label"] == p["vanilla_label"]
    )
    record = {
        "algorithm": args.algorithm,
        "testbed": args.testbed,
        "run_name": run_name,
        "partition_index": int(partition_index),
        "chunk_index": int(chunk_index),
        "partition_seed": int(partition_seed),
        "chunk_consistency_ids": chunk_cids,
        "n_items": len(items),
        "n_predicted": int(n_predicted),
        "duration_seconds": float(duration),
        "cache_key_args": cache_args,
        "eval_set_meta": {
            k: eval_set_meta[k]
            for k in ("testbed", "source_file", "n_consistency_ids", "seed")
            if k in eval_set_meta
        },
        "final_metric_summary": {
            "train_accuracy": float(final_metric["train_accuracy"]),
            "train_size": int(final_metric["train_size"]),
            "predict_distribution": dict(final_metric["train_predict_distribution"]),
            "label_distribution": dict(final_metric["train_label_distribution"]),
        },
        "chunk_accuracy": float(n_correct / n_predicted) if n_predicted else None,
        "predictions": predictions,
    }
    with cache_path.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    _chunk_log(
        f"[p{partition_index:02d} c{chunk_index:02d}] done in {duration:.1f}s, "
        f"acc={record['chunk_accuracy']}"
    )
    return record


def run_gibbs_chunk(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    parallel = getattr(args, "num_workers", 1) > 1
    return _run_label_predict_chunk(
        search_fn=run_gibbs_search,
        verbose_steps=not parallel,
        **kwargs,
    )


def run_barker_gibbs_chunk(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    parallel = getattr(args, "num_workers", 1) > 1
    return _run_label_predict_chunk(
        search_fn=run_barker_gibbs_search,
        verbose_steps=not parallel,
        **kwargs,
    )


def run_gambling_gibbs_chunk(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    parallel = getattr(args, "num_workers", 1) > 1
    return _run_label_predict_chunk(
        search_fn=run_gambling_gibbs_search,
        verbose_steps=not parallel,
        **kwargs,
    )


def run_zeroshot_chunk(**kwargs: Any) -> dict[str, Any]:
    return _run_label_predict_chunk(search_fn=run_zeroshot_search, verbose_steps=False, **kwargs)


def run_npass_chunk(**kwargs: Any) -> dict[str, Any]:
    return _run_label_predict_chunk(search_fn=run_npass_search, verbose_steps=False, **kwargs)


def _run_single_chunk(
    chunk_runner: Callable[..., dict[str, Any]],
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    partition_index: int,
    partition_seed: int,
    chunk_index: int,
    output_dir: Path,
    eval_set_meta: dict[str, Any],
    model_api: Any,
    run_name: str,
) -> dict[str, Any]:
    return chunk_runner(
        args=args,
        items=items,
        partition_index=partition_index,
        partition_seed=partition_seed,
        chunk_index=chunk_index,
        output_dir=output_dir,
        eval_set_meta=eval_set_meta,
        model_api=model_api,
        run_name=run_name,
    )


def _run_chunks_parallel(
    chunk_runner: Callable[..., dict[str, Any]],
    *,
    args: argparse.Namespace,
    jobs: list[tuple[int, int, list[dict[str, Any]], int]],
    output_dir: Path,
    eval_set_meta: dict[str, Any],
    model_api: Any,
    run_name: str,
) -> None:
    num_workers = args.num_workers
    _chunk_log(f"[parallel] running {len(jobs)} chunks with {num_workers} workers")

    failures: list[tuple[int, int, BaseException]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_job = {
            executor.submit(
                _run_single_chunk,
                chunk_runner,
                args=args,
                items=items,
                partition_index=partition_index,
                partition_seed=partition_seed,
                chunk_index=chunk_index,
                output_dir=output_dir,
                eval_set_meta=eval_set_meta,
                model_api=model_api,
                run_name=run_name,
            ): (partition_index, chunk_index)
            for partition_index, chunk_index, items, partition_seed in jobs
        }
        for future in as_completed(future_to_job):
            partition_index, chunk_index = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((partition_index, chunk_index, exc))

    if failures:
        details = "\n".join(f"  p{p:02d} c{c:02d}: {exc!r}" for p, c, exc in sorted(failures))
        raise RuntimeError(f"{len(failures)} chunk(s) failed under parallel execution:\n{details}")


# --- Aggregation ------------------------------------------------------------


def _summarize_partition(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_correct = 0
    n_predicted = 0
    n_missing = 0
    for r in records:
        for p in r["predictions"]:
            if p["predicted_label"] is None:
                n_missing += 1
                continue
            n_predicted += 1
            if p["predicted_label"] == p["vanilla_label"]:
                n_correct += 1
    return {
        "n_predicted": n_predicted,
        "n_missing": n_missing,
        "accuracy": float(n_correct / n_predicted) if n_predicted else None,
    }


def aggregate_run(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    run_name: str,
    eval_set_meta: dict[str, Any],
    partitions: list[dict[str, Any]],
) -> dict[str, Any]:
    per_partition_records: list[list[dict[str, Any]]] = []
    for p_info in partitions:
        p_records = []
        for c in range(len(p_info["chunks"])):
            cache_path = _chunk_cache_path(output_dir, p_info["partition_index"], c)
            with cache_path.open() as f:
                p_records.append(json.load(f))
        per_partition_records.append(p_records)

    per_partition_summaries = [_summarize_partition(prs) for prs in per_partition_records]
    accs = [s["accuracy"] for s in per_partition_summaries if s["accuracy"] is not None]
    headline = {
        "n_partitions": len(accs),
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "std_accuracy": float(np.std(accs)) if accs else None,
        "se_accuracy": float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else None,
        "min_accuracy": float(np.min(accs)) if accs else None,
        "max_accuracy": float(np.max(accs)) if accs else None,
    }

    per_item_correct_count: dict[int, int] = {}
    per_item_pred_count: dict[int, int] = {}
    per_item_truth: dict[int, int] = {}
    for prs in per_partition_records:
        for r in prs:
            for p in r["predictions"]:
                src = int(p["source_index"])
                per_item_truth[src] = int(p["vanilla_label"])
                if p["predicted_label"] is None:
                    continue
                per_item_pred_count[src] = per_item_pred_count.get(src, 0) + 1
                if p["predicted_label"] == p["vanilla_label"]:
                    per_item_correct_count[src] = per_item_correct_count.get(src, 0) + 1

    n_partitions = len(partitions)
    correct_count_dist = Counter(per_item_correct_count.get(src, 0) for src in per_item_truth)
    summary = {
        "algorithm": args.algorithm,
        "testbed": args.testbed,
        "run_name": run_name,
        "n_partitions": n_partitions,
        "partition_base_seed": args.partition_base_seed,
        "chunk_size_cis": args.chunk_size_cis,
        "n_chunks_per_partition": len(partitions[0]["chunks"]),
        "eval_set_meta": {
            k: eval_set_meta[k]
            for k in (
                "testbed",
                "source_file",
                "n_consistency_ids",
                "n_items",
                "seed",
            )
            if k in eval_set_meta
        },
        "headline": headline,
        "per_partition_accuracy": [s["accuracy"] for s in per_partition_summaries],
        "per_partition_summary": per_partition_summaries,
        "per_item_correct_count_distribution": {
            str(k): correct_count_dist.get(k, 0) for k in range(n_partitions + 1)
        },
        "n_items_total": len(per_item_truth),
        "cache_key_args": relevant_args_snapshot(args),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {summary_path}")

    def _fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    print(
        f"headline: mean={_fmt(headline['mean_accuracy'])} "
        f"+/- {_fmt(headline['se_accuracy'])} (SE) "
        f"over {headline['n_partitions']} partitions, "
        f"min={_fmt(headline['min_accuracy'])}, max={_fmt(headline['max_accuracy'])}"
    )
    print(
        f"per-item correct-count dist (out of {n_partitions}): "
        f"{summary['per_item_correct_count_distribution']}"
    )
    return summary


def _print_plan(
    *,
    eval_set: dict[str, Any],
    partitions: list[dict[str, Any]],
    cid_index: dict[Any, list[int]],
    args: argparse.Namespace,
) -> None:
    n_chunks = len(partitions[0]["chunks"])
    chunk_size_items = args.chunk_size_cis * eval_set["group_size"]
    total_chunks = sum(len(p["chunks"]) for p in partitions)
    total_items = total_chunks * chunk_size_items
    missing = [cid for cid in eval_set["consistency_ids"] if cid not in cid_index]
    print("=== run plan ===")
    print(f"  algorithm           : {args.algorithm}")
    print(f"  testbed             : {args.testbed}")
    print(f"  eval_set            : {args.eval_set}")
    print(
        f"  CIs in eval set     : {len(eval_set['consistency_ids'])} "
        f"(group_size={eval_set['group_size']}, items={eval_set['n_items']})"
    )
    print(f"  n_partitions        : {len(partitions)}")
    print(f"  partition_base_seed : {args.partition_base_seed}")
    print(f"  chunk_size_cis      : {args.chunk_size_cis} ({chunk_size_items} items)")
    print(f"  chunks per partition: {n_chunks}")
    print(f"  total chunk runs    : {total_chunks}")
    print(f"  total item-preds    : {total_items}")
    print(f"  missing CIs in src  : {len(missing)} (sample: {missing[:5]})")
    print(f"  output_dir          : {args.output_dir}")
    print(f"  run_name            : {args.run_name}")
    print(f"  num_workers         : {getattr(args, 'num_workers', 1)}")
    print(f"  cache_key_args      : {relevant_args_snapshot(args)}")
    for p in partitions[: min(2, len(partitions))]:
        sample_chunks = [c[:5] for c in p["chunks"][:3]]
        print(
            f"  partition {p['partition_index']:02d} (seed={p['partition_seed']}): "
            f"first chunks (first 5 CIs) = {sample_chunks}"
        )


def main(args: argparse.Namespace) -> None:
    setup_environment(logger_level="error")

    eval_set = load_eval_set(args.eval_set)
    if eval_set["testbed"] != args.testbed:
        raise ValueError(
            f"eval_set testbed={eval_set['testbed']!r} does not match "
            f"--testbed={args.testbed!r}"
        )

    train = load_processed_train(args)  # also sets args.GROUP_SIZE
    if args.GROUP_SIZE != eval_set["group_size"]:
        raise ValueError(
            f"GROUP_SIZE for testbed={args.testbed} is {args.GROUP_SIZE}, "
            f"but eval set was built with group_size={eval_set['group_size']}."
        )
    cid_index = build_cid_index(train)

    partitions = build_partitions(
        eval_set["consistency_ids"],
        n_partitions=args.n_partitions,
        base_seed=args.partition_base_seed,
        chunk_size_cis=args.chunk_size_cis,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "GROUP_SIZE": args.GROUP_SIZE,
                "cache_key_args": relevant_args_snapshot(args),
            },
            f,
            indent=2,
            default=str,
        )

    _print_plan(eval_set=eval_set, partitions=partitions, cid_index=cid_index, args=args)

    if args.algorithm == "icm":
        model_client: Any = ModelAPI(anthropic_num_threads=20, openai_fraction_rate_limit=0.99)
    elif args.algorithm in ("gibbs", "zeroshot", "npass", "barker_gibbs", "gambling_gibbs"):
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

        if args.algorithm == "barker_gibbs":
            max_tokens = 5
        elif args.algorithm == "gambling_gibbs":
            if getattr(args, "manual_reasoning", False):
                max_tokens = 1024
            else:
                max_tokens = 10
        else:
            max_tokens = 2

        client_cls = (
            ReasoningOpenAICompatLLM if args.algorithm == "gambling_gibbs" else OpenAICompatLLM
        )
        model_client = client_cls(
            model_name=args.model,
            base_url=f"http://localhost:{args.port}/v1",
            system_prompt=getattr(args, "system_prompt", ""),
            temperature=args.temperature,
            max_tokens=max_tokens,
            instruction_tuned=args.instruction_tuned,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    selected_partitions = (
        partitions
        if args.only_partition is None
        else [p for p in partitions if p["partition_index"] == args.only_partition]
    )
    if not selected_partitions:
        raise ValueError(f"No partition index {args.only_partition} in plan.")

    chunk_runner = {
        "icm": run_icm_chunk,
        "gibbs": run_gibbs_chunk,
        "barker_gibbs": run_barker_gibbs_chunk,
        "gambling_gibbs": run_gambling_gibbs_chunk,
        "zeroshot": run_zeroshot_chunk,
        "npass": run_npass_chunk,
    }[args.algorithm]

    chunk_jobs: list[tuple[int, int, list[dict[str, Any]], int]] = []
    for p_info in selected_partitions:
        p = p_info["partition_index"]
        for c, chunk_cids in enumerate(p_info["chunks"]):
            items = select_items_for_chunk(
                train, cid_index, chunk_cids, expected_group_size=args.GROUP_SIZE
            )
            chunk_jobs.append((p, c, items, p_info["partition_seed"]))

    if args.num_workers <= 1:
        for partition_index, chunk_index, items, partition_seed in chunk_jobs:
            _run_single_chunk(
                chunk_runner,
                args=args,
                items=items,
                partition_index=partition_index,
                partition_seed=partition_seed,
                chunk_index=chunk_index,
                output_dir=args.output_dir,
                eval_set_meta=eval_set,
                model_api=model_client,
                run_name=args.run_name,
            )
    else:
        _run_chunks_parallel(
            chunk_runner,
            args=args,
            jobs=chunk_jobs,
            output_dir=args.output_dir,
            eval_set_meta=eval_set,
            model_api=model_client,
            run_name=args.run_name,
        )

    if args.skip_aggregate:
        print("\n[skip_aggregate] not aggregating.")
        return

    missing = [
        (p_info["partition_index"], c)
        for p_info in partitions
        for c in range(len(p_info["chunks"]))
        if not _chunk_cache_path(args.output_dir, p_info["partition_index"], c).exists()
    ]
    if missing:
        print(
            f"\n[aggregate] skipping: {len(missing)} chunks have no cache file. "
            f"First few: {missing[:5]}"
        )
        return

    aggregate_run(
        args=args,
        output_dir=args.output_dir,
        run_name=args.run_name,
        eval_set_meta=eval_set,
        partitions=partitions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--algorithm",
        choices=["icm", "gibbs", "zeroshot", "npass", "barker_gibbs", "gambling_gibbs"],
        default="icm",
    )
    parser.add_argument(
        "--testbed",
        choices=["alpaca", "gsm8k", "truthfulQA", "truthfulQA-preference"],
        required=True,
    )
    parser.add_argument(
        "--eval_set",
        type=Path,
        default=None,
        help="Path to eval-set JSON.",
    )

    parser.add_argument("--n_partitions", type=int, default=5)
    parser.add_argument("--partition_base_seed", type=int, default=42)
    parser.add_argument(
        "--chunk_size_cis",
        type=int,
        default=16,
        help="(icm/gibbs) Number of consistency_ids per chunk.",
    )

    # Algorithm hyperparams (ICM)
    parser.add_argument("--K", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="log")

    # Algorithm hyperparams (Gibbs)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="(gibbs) Bernoulli temperature applied to the LLM log-odds.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=25,
        help="(gibbs) Number of retained samples.",
    )

    parser.add_argument(
        "--system_prompt", type=str, default=None, help="System prompt for instruction-tuned models"
    )
    parser.add_argument(
        "--use_new_system_prompt",
        action="store_true",
        help="Use the new system prompt for the model.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        default=None,
        help="(gibbs) Number of Gibbs steps to discard before collecting samples.",
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=None,
        help="(gibbs) Number of Gibbs steps between retained samples.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="(gibbs) Use a systematic sweep.",
    )
    parser.add_argument(
        "--manual_reasoning",
        action="store_true",
        help="(gambling_gibbs) Use step-by-step reasoning.",
    )

    # Algorithm hyperparams (NPass)
    parser.add_argument(
        "--n_passes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--all_passes",
        action="store_true",
    )

    parser.add_argument(
        "--no_trailing_space",
        action="store_true",
    )

    # Model
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--port", type=int, default=8000)

    # Output / control
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write results to.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run.",
    )
    parser.add_argument(
        "--only_partition",
        type=int,
        default=None,
        help="If set, run only this partition index.",
    )
    parser.add_argument(
        "--skip_aggregate",
        action="store_true",
        help="Skip aggregation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of chunks to run concurrently.",
    )

    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError(f"--num_workers must be >= 1, got {args.num_workers}.")

    if args.eval_set is None:
        args.eval_set = CONSISTENT_REASONING_DIR / "eval_sets" / f"{args.testbed}.json"

    args.instruction_tuned = False
    if args.model in INSTRUCT_MODELS:
        args.instruction_tuned = True

    if args.instruction_tuned:
        args.no_trailing_space = False

    if args.system_prompt is None and args.instruction_tuned:
        args.system_prompt = "You are a helpful assistant."
        if args.use_new_system_prompt:
            args.system_prompt = (
                "You are a helpful assistant. You verify whether a claim correctly "
                "answers a question: True if the claim is correct, False if not."
            )

    if args.algorithm in ("zeroshot", "npass") and args.chunk_size_cis != 1:
        print(f"[run_eval] {args.algorithm} ignores chunk_size_cis; forcing chunk_size_cis=1.")
        args.chunk_size_cis = 1

    if args.algorithm in ("gibbs", "barker_gibbs", "gambling_gibbs"):
        _GROUP_SIZES = {"alpaca": 2, "gsm8k": 4, "truthfulQA": 4, "truthfulQA-preference": 2}
        chunk_size_items = args.chunk_size_cis * _GROUP_SIZES[args.testbed]
        if args.thinning is None:
            args.thinning = chunk_size_items
        if args.burn_in is None:
            args.burn_in = min(10, args.num_samples) * args.thinning

    def derive_run_name(args: argparse.Namespace) -> str:
        model_short = args.model.split("/")[-1]

        system_prompt_str = ""
        if args.instruction_tuned:
            if args.use_new_system_prompt:
                system_prompt_str = "NewSysP_"

        if args.algorithm == "icm":
            nots = "_nots" if args.no_trailing_space else ""
            return (
                f"{system_prompt_str}icm_{model_short}"
                f"_K{args.K}_a{args.alpha}_iT{args.initial_T}_fT{args.final_T}"
                f"_{args.scheduler}_decay{args.decay}_ns{args.num_seed}{nots}"
                f"_cs{args.chunk_size_cis}"
                f"_baseseed{args.partition_base_seed}"
            )
        if args.algorithm in ("gibbs", "barker_gibbs", "gambling_gibbs"):
            scan = "_sweep" if args.sweep else ""
            nots = "_nots" if args.no_trailing_space else ""
            reasoning = "_reasoning" if getattr(args, "manual_reasoning", False) else ""
            if args.algorithm == "gibbs":
                prefix = "gibbs_"
            elif args.algorithm == "barker_gibbs":
                prefix = "barkergibbs_"
            else:
                prefix = "gamblinggibbs_"
            return (
                f"{system_prompt_str}{prefix}{model_short}"
                f"_T{args.temperature}_burn{args.burn_in}_thin{args.thinning}"
                f"_K{args.num_samples}{scan}{nots}{reasoning}_cs{args.chunk_size_cis}"
                f"_baseseed{args.partition_base_seed}"
            )
        if args.algorithm == "zeroshot":
            nots = "_nots" if args.no_trailing_space else ""
            return (
                f"{system_prompt_str}zeroshot_{model_short}"
                f"_T{args.temperature}{nots}"
                f"_baseseed{args.partition_base_seed}"
            )
        if args.algorithm == "npass":
            nots = "_nots" if args.no_trailing_space else ""
            n_str = "Nall" if args.all_passes else f"N{args.n_passes}"
            return (
                f"{system_prompt_str}npass_{model_short}"
                f"_T{args.temperature}_{n_str}{nots}"
                f"_baseseed{args.partition_base_seed}"
            )
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    if args.run_name is None:
        args.run_name = derive_run_name(args)

    if args.output_dir is None:
        args.output_dir = (
            CONSISTENT_REASONING_DIR
            / "eval_results"
            / args.algorithm
            / args.testbed
            / args.run_name
        )

    os.environ["LLAMA_API_BASE"] = f"http://localhost:{args.port}/v1"
    main(args)
