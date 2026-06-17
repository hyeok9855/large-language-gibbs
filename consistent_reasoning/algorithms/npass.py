from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot

_LABEL_CHOICES: list[str] = ["True", "False"]


def _enumerate_permutations(uids: list[int], n_passes: int) -> list[list[int]]:
    g = len(uids)
    max_n = math.factorial(g)
    n = min(n_passes, max_n)

    if n_passes >= max_n:
        all_perms = [list(p) for p in itertools.permutations(uids)]
        random.shuffle(all_perms)
        return all_perms

    seen: set[tuple[int, ...]] = set()
    perms: list[list[int]] = []
    while len(perms) < n:
        candidate = list(uids)
        random.shuffle(candidate)
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        perms.append(candidate)
    return perms


def run_npass_search(
    demonstrations: dict[int, dict[str, Any]],
    whole_ids: list[int],
    args: argparse.Namespace,
    llm: OpenAICompatLLM,
    log_path: Path | str | None,
    *,
    verbose_steps: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    no_trailing_space = bool(getattr(args, "no_trailing_space", False))
    label_choices: list[str] = [" True", " False"] if no_trailing_space else list(_LABEL_CHOICES)
    all_passes = bool(getattr(args, "all_passes", False))
    n_passes = int(getattr(args, "n_passes", 4))
    if not all_passes and n_passes < 1:
        raise ValueError(f"n_passes must be >= 1 (got {n_passes}) unless all_passes=True")

    cid_to_uids: dict[Any, list[int]] = defaultdict(list)
    for uid in whole_ids:
        cid = demonstrations[uid]["consistency_id"]
        cid_to_uids[cid].append(uid)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    def _effective_n_passes(group_size: int) -> int:
        if all_passes:
            return math.factorial(group_size)
        return min(n_passes, math.factorial(group_size))

    print(
        f"[npass]: {len(cid_to_uids)} consistency groups "
        f"({'all_passes' if all_passes else f'n_passes={n_passes}'}, "
        f"T={llm.temperature}, no_trailing_space={no_trailing_space})"
    )

    true_counts: dict[int, int] = {uid: 0 for uid in whole_ids}
    pred_counts: dict[int, int] = {uid: 0 for uid in whole_ids}

    total_queries = sum(_effective_n_passes(len(uids)) * len(uids) for uids in cid_to_uids.values())
    pbar = tqdm(total=total_queries, desc="npass queries")

    for cid, uids in cid_to_uids.items():
        effective_n = _effective_n_passes(len(uids))
        permutations = _enumerate_permutations(uids, effective_n)
        for perm_idx, perm in enumerate(permutations):
            history: list[dict[str, Any]] = []
            for position, uid in enumerate(perm):
                example = demonstrations[uid]
                if getattr(args, "instruction_tuned", False):
                    prompt = example["prompt"]
                    chosen = llm.generate(prompt, schema=label_choices, verbose=False, history=history)
                else:
                    prompt = cast(str, get_judge_prompt_fewshot(example, history, pipeline=False))
                    if no_trailing_space and prompt.endswith(" "):
                        prompt = prompt[:-1]
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

                if verbose_steps:
                    print(
                        f"[npass] cid={cid} perm={perm_idx} pos={position} "
                        f"uid={uid} -> {bool(value)}"
                    )
                pbar.update(1)
    pbar.close()

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
    if n_ties:
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
