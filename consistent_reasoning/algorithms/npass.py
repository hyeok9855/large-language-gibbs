from __future__ import annotations

import argparse
import itertools
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


def _enumerate_permutations(uids: list[int], all_pass: bool, n_passes: int) -> list[list[int]]:
    if all_pass:
        assert len(uids) <= 5, "All-pass mode is only supported for groups of size <= 5."
        all_perms = [list(p) for p in itertools.permutations(uids)]
        random.shuffle(all_perms)
        return all_perms

    # If not all-pass, sample n_passes permutations with replacement.
    perms: list[list[int]] = []
    while len(perms) < n_passes:
        perm = list(uids)
        random.shuffle(perm)
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
    all_pass = bool(args.all_pass)
    n_passes = int(args.n_passes)  # When all_pass=True, n_passes is ignored.

    cid_to_uids: dict[Any, list[int]] = defaultdict(list)
    for uid in whole_ids:
        cid = demonstrations[uid]["consistency_id"]
        cid_to_uids[cid].append(uid)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    if verbose:
        print(
            f"[npass]: {len(cid_to_uids)} consistency groups "
            f"({'all_pass' if all_pass else f'n_passes={n_passes}'}, T={llm.temperature})"
        )

    true_counts: dict[int, int] = {uid: 0 for uid in whole_ids}
    pred_counts: dict[int, int] = {uid: 0 for uid in whole_ids}

    parallel = getattr(args, "num_workers", 1) > 1
    for cid, uids in tqdm(cid_to_uids.items(), desc="npass", disable=parallel):
        permutations = _enumerate_permutations(uids, all_pass, n_passes)
        for perm_idx, perm in enumerate(permutations):
            history: list[dict[str, Any]] = []
            for position, uid in enumerate(perm):
                example = demonstrations[uid]
                if args.instruction_tuned:
                    prompt = example["prompt"]
                    chosen = llm.generate(
                        prompt, schema=label_choices, verbose=False, history=history
                    )
                else:
                    prompt = cast(str, get_judge_prompt_fewshot(example, history, pipeline=False))
                    chosen = llm.generate(prompt, schema=label_choices, verbose=False)
                if not isinstance(chosen, str):
                    raise TypeError(
                        f"Expected a string from choice-constrained generation; "
                        f"got {type(chosen)}"
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
                        f"[npass] cid={cid} perm={perm_idx} pos={position} "
                        f"uid={uid} -> {bool(value)}"
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
