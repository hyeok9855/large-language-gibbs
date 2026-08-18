from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot


def _enumerate_permutations(
    cid_to_uids: dict[Any, list[int]], n_passes: int, fixed_order: bool = False
) -> list[list[int]]:
    if fixed_order:
        # Every pass uses the chunk's original item order; variation across
        # passes then comes only from label sampling at T > 0.
        perm = [uid for uids in cid_to_uids.values() for uid in uids]
        return [list(perm) for _ in range(n_passes)]

    # Sample n_passes block-preserving permutations with replacement.
    perms: list[list[int]] = []
    while len(perms) < n_passes:
        cids = list(cid_to_uids.keys())
        random.shuffle(cids)
        perm: list[int] = []
        for cid in cids:
            members = list(cid_to_uids[cid])
            random.shuffle(members)
            perm.extend(members)
        perms.append(perm)
    return perms


def run_npass_search(
    demonstrations: dict[int, dict[str, Any]],
    whole_ids: list[int],
    args: argparse.Namespace,
    llm: OpenAICompatLLM,
    log_path: Path | str | None,
    *,
    verbose: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    label_choices: list[str] = ["True", "False"] if args.instruction_tuned else [" True", " False"]
    n_passes = int(args.n_passes)

    cid_to_uids: dict[Any, list[int]] = defaultdict(list)
    for uid in whole_ids:
        cid_to_uids[demonstrations[uid]["consistency_id"]].append(uid)
    n_cids = len(cid_to_uids)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    if verbose:
        print(
            f"[npass]: pool of {len(whole_ids)} items across {n_cids} consistency "
            f"group(s) (n_passes={n_passes}, fixed_order={args.fixed_order}, T={llm.temperature})"
        )

    true_counts: dict[int, int] = {uid: 0 for uid in whole_ids}
    pred_counts: dict[int, int] = {uid: 0 for uid in whole_ids}

    parallel = args.num_workers > 1
    permutations = _enumerate_permutations(cid_to_uids, n_passes, args.fixed_order)
    for perm_idx, perm in enumerate(tqdm(permutations, desc="npass", disable=parallel)):
        history: list[dict[str, Any]] = []
        for position, uid in enumerate(perm):
            example = demonstrations[uid]
            cid = example["consistency_id"]
            if args.instruction_tuned:
                prompt = example["prompt"]
                chosen = llm.generate(prompt, schema=label_choices, verbose=False, history=history)
            else:
                prompt = cast(str, get_judge_prompt_fewshot(example, history, pipeline=False))
                chosen = llm.generate(prompt, schema=label_choices, verbose=False)
            if not isinstance(chosen, str):
                raise TypeError(
                    f"Expected a string from choice-constrained generation; got {type(chosen)}"
                )
            value = chosen.strip().capitalize() == "True"

            true_counts[uid] += int(value)
            pred_counts[uid] += 1

            history_item = deepcopy(example)
            history_item["label"] = int(value)
            history.append(history_item)

            if log_path is not None:
                log_record = {
                    "consistency_id": cid,
                    "perm_index": int(perm_idx),
                    "position": int(position),
                    "uid": int(uid),
                    "chosen": chosen,
                    "label": int(value),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_record, default=str) + "\n")

            if verbose:
                print(
                    f"[npass] perm={perm_idx} pos={position} cid={cid} uid={uid} -> {bool(value)}"
                )

    final_demos = deepcopy(demonstrations)
    n_ties = 0
    for uid in whole_ids:
        n = pred_counts[uid]
        if n == 0:
            final_demos[uid]["label"] = None
            final_demos[uid]["_predicted_score"] = None
            continue
        score = true_counts[uid] / n
        if 2 * true_counts[uid] == n:
            label = random.randint(0, 1)
            n_ties += 1
        else:
            label = 1 if score > 0.5 else 0
        final_demos[uid]["label"] = int(label)
        final_demos[uid]["_predicted_score"] = float(score)

    if n_ties and verbose:
        print(f"[npass]: {n_ties}/{len(whole_ids)} items broken at random (tied 50/50)")

    final_metric = {
        "train_accuracy": float(
            np.mean(
                [
                    v["label"] == v["vanilla_label"]
                    for v in final_demos.values()
                    if v.get("label") is not None
                ]
            )
        ),
        "train_predict_distribution": dict(Counter(v.get("label") for v in final_demos.values())),
        "train_label_distribution": dict(Counter(v["vanilla_label"] for v in final_demos.values())),
        "train_size": len(final_demos),
    }
    return final_demos, final_metric
