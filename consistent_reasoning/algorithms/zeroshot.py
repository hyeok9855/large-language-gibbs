from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot

_LABEL_CHOICES: list[str] = ["True", "False"]


def run_zeroshot_search(
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

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    print(
        f"[zeroshot]: N={len(whole_ids)} items "
        f"(T={llm.temperature}, no_trailing_space={no_trailing_space})"
    )

    final_demos = deepcopy(demonstrations)
    for uid in tqdm(whole_ids, desc="zeroshot"):
        example = final_demos[uid]
        prompt = cast(str, get_judge_prompt_fewshot(example, [], pipeline=False))
        if no_trailing_space and prompt.endswith(" "):
            prompt = prompt[:-1]

        chosen = llm.generate(prompt, schema=label_choices, verbose=False)
        if not isinstance(chosen, str):
            raise TypeError(
                f"Expected a string from choice-constrained generation; got {type(chosen)}"
            )
        value = chosen.strip().capitalize() == "True"
        example["label"] = int(value)
        example["_predicted_score"] = 1.0 if value else 0.0

        if log_path is not None:
            log_record = {
                "uid": int(uid),
                "consistency_id": example.get("consistency_id"),
                "vanilla_label": int(bool(example["vanilla_label"])),
                "chosen": chosen,
                "label": int(value),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_record, default=str) + "\n")

        if verbose_steps:
            print(
                f"[zeroshot] uid={uid} -> {bool(value)} "
                f"(vanilla={int(bool(example['vanilla_label']))})"
            )

    final_metric = {
        "train_accuracy": float(
            np.mean([v["label"] == v["vanilla_label"] for v in final_demos.values()])
        ),
        "train_predict_distribution": dict(Counter(v["label"] for v in final_demos.values())),
        "train_label_distribution": dict(Counter(v["vanilla_label"] for v in final_demos.values())),
        "train_size": len(final_demos),
    }
    return final_demos, final_metric
