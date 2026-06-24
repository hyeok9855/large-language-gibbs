"""Build deterministic evaluation sets of consistency_ids per testbed.

Each eval set is a JSON file describing a fixed-size, seeded subsample of
consistency groups from a source dataset. Algorithms (ICM, Gibbs, ...) load
this file, resolve the listed consistency_ids against the source file, and
report per-item accuracy on the resulting fixed item set so that runs are
comparable across methods, seeds, and machines.

Usage:
    cd Unsupervised-Elicitation
    python eval_sets/build_eval_sets.py            # builds every configured eval set
    python eval_sets/build_eval_sets.py --testbed truthfulQA
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


# Each entry describes how to construct one eval set. The output JSON stores
# enough metadata that a downstream loader can reconstruct the exact items
# without consulting this file.
EVAL_SET_CONFIGS: dict[str, dict[str, Any]] = {
    "truthfulQA": {
        "source_file": "data/truthfulqa.json",
        "label_field": "label",
        "choice_field": "choice",
        "group_size": 4,
        "n_consistency_ids": 256,
        "seed": 42,
    },
    "gsm8k": {
        "source_file": "data/gsm8k.json",
        "label_field": "label",
        "choice_field": "choice",
        "group_size": 4,
        "n_consistency_ids": 256,
        "seed": 42,
    },
    "alpaca": {
        "source_file": "data/alpaca.json",
        "label_field": "label",
        "choice_field": "choice",
        "group_size": 2,  # alpaca groups are pairs, not quads
        "n_consistency_ids": 256,
        "seed": 42,
    },
}


REPO_ROOT = Path(__file__).resolve().parent.parent  # Unsupervised-Elicitation/
OUTPUT_DIR = REPO_ROOT / "eval_sets"


def _load_dataset(source_file: Path) -> list[dict[str, Any]]:
    with source_file.open() as f:
        return json.load(f)


def _group_by_consistency_id(items: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["consistency_id"], []).append(item)
    return groups


def build_eval_set(testbed: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return the eval-set dict for a single testbed configuration."""
    source_path = REPO_ROOT / config["source_file"]
    items = _load_dataset(source_path)
    groups = _group_by_consistency_id(items)

    expected_size = config["group_size"]
    bad_groups = [cid for cid, members in groups.items() if len(members) != expected_size]
    if bad_groups:
        raise ValueError(
            f"{testbed}: {len(bad_groups)} consistency groups in {source_path} "
            f"do not have the expected group size {expected_size}; "
            f"first few = {bad_groups[:5]}"
        )

    n_groups_available = len(groups)
    n_to_sample = config["n_consistency_ids"]
    if n_to_sample > n_groups_available:
        raise ValueError(
            f"{testbed}: requested {n_to_sample} consistency_ids but only "
            f"{n_groups_available} are available in {source_path}."
        )

    rng = random.Random(config["seed"])
    all_cids = sorted(groups.keys())
    sampled_cids = sorted(rng.sample(all_cids, n_to_sample))

    selected_items: list[dict[str, Any]] = []
    for cid in sampled_cids:
        selected_items.extend(groups[cid])
    label_field = config["label_field"]
    label_dist = Counter(int(bool(it[label_field])) for it in selected_items)
    true_per_cid = Counter(
        sum(int(bool(it[label_field])) for it in groups[cid]) for cid in sampled_cids
    )

    return {
        "testbed": testbed,
        "source_file": config["source_file"],
        "label_field": label_field,
        "choice_field": config["choice_field"],
        "group_size": expected_size,
        "n_consistency_ids": n_to_sample,
        "n_items": n_to_sample * expected_size,
        "seed": config["seed"],
        "n_groups_available": n_groups_available,
        "label_distribution": dict(label_dist),
        "true_per_consistency_id": dict(sorted(true_per_cid.items())),
        "consistency_ids": sampled_cids,
    }


def write_eval_set(eval_set: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{eval_set['testbed']}.json"
    with output_path.open("w") as f:
        json.dump(eval_set, f, indent=2)
        f.write("\n")
    return output_path


def main(args: argparse.Namespace) -> None:
    selected = (
        list(EVAL_SET_CONFIGS.items())
        if args.testbed == "all"
        else [(args.testbed, EVAL_SET_CONFIGS[args.testbed])]
    )

    for testbed, config in selected:
        eval_set = build_eval_set(testbed, config)
        output_path = write_eval_set(eval_set, OUTPUT_DIR)
        rel = output_path.relative_to(REPO_ROOT)
        print(
            f"[{testbed}] wrote {rel}: "
            f"{eval_set['n_consistency_ids']} CIs, "
            f"{eval_set['n_items']} items, "
            f"label distribution={eval_set['label_distribution']}, "
            f"#True per CI={eval_set['true_per_consistency_id']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testbed",
        choices=[*EVAL_SET_CONFIGS.keys(), "all"],
        default="all",
    )
    args = parser.parse_args()

    main(args)
