from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from tqdm import tqdm

from priorbot.priors import GibbsLLMPrior, Prior
from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot


class CustomPrior(Prior):
    def __init__(
        self,
        llm: OpenAICompatLLM,
        demonstrations: dict[int, dict[str, Any]],
    ):
        super().__init__()
        self.llm = llm
        self.demonstrations = demonstrations
        self._label_choices = (
            ["True", "False"] if self.llm.instruction_tuned else [" True", " False"]
        )

    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        keys = list(schema["properties"].keys())
        n = len(keys)
        n_true = (n + 1) // 2
        samples: list[dict[str, Any]] = []
        for _ in range(n_samples):
            labels = [True] * n_true + [False] * (n - n_true)
            random.shuffle(labels)
            samples.append(dict(zip(keys, labels)))
        return samples

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        if len(schema["properties"]) != 1:
            raise ValueError(
                "CustomPrior.sample_conditional expects a single-key schema; "
                f"got {list(schema['properties'])}."
            )

        (key,) = schema["properties"].keys()
        uid = int(key)
        example = self.demonstrations[uid]
        current_pool = self._build_pool_from_observed(observed, skip_uid=uid)
        demos = _hierarchical_demo_order(
            current_pool,
            target_consistency_id=example["consistency_id"],
            target_uid=example["uid"],
        )
        if self.llm.instruction_tuned:
            prompt = example["prompt"]
            chosen = self.llm.generate(
                prompt, schema=self._label_choices, verbose=verbose, history=demos
            )
        else:
            prompt = cast(str, get_judge_prompt_fewshot(example, demos, pipeline=False))
            chosen = self.llm.generate(prompt, schema=self._label_choices, verbose=verbose)

        if not isinstance(chosen, str):
            raise TypeError(
                f"Expected a string from choice-constrained generation; got {type(chosen)}"
            )
        value = chosen.strip().capitalize() == "True"

        if verbose:
            print(f"[Gibbs] uid={uid} -> {value} (chosen={chosen!r})")

        return [{key: value} for _ in range(n_samples)]

    def _build_pool_from_observed(
        self,
        observed: dict[str, Any],
        skip_uid: int,
    ) -> dict[int, dict[str, Any]]:
        pool: dict[int, dict[str, Any]] = {}
        for key, value in observed.items():
            uid = int(key)
            if uid == skip_uid:
                continue

            example = deepcopy(self.demonstrations[uid])
            example["label"] = int(bool(value))
            pool[uid] = example
        return pool


def _hierarchical_demo_order(
    current_pool: dict[int, dict[str, Any]],
    target_consistency_id: Any,
    target_uid: int,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    target_group: list[dict[str, Any]] = []

    for uid, example in current_pool.items():
        if example.get("label") is None or uid == target_uid:
            continue

        if example["consistency_id"] == target_consistency_id:
            target_group.append(example)
        else:
            groups.setdefault(example["consistency_id"], []).append(example)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    ordered: list[dict[str, Any]] = []
    for group_key in group_keys:
        members = list(groups[group_key])
        random.shuffle(members)
        ordered.extend(members)

    random.shuffle(target_group)
    ordered.extend(target_group)
    return ordered


class LoggedGibbsLLMPrior(GibbsLLMPrior):
    def __init__(
        self,
        base_prior: Prior,
        burn_in: int,
        thinning: int,
        sweep: bool = False,
        on_step: Callable[[int, dict[str, Any], str], None] | None = None,
    ):
        super().__init__(base_prior, burn_in, thinning, block_size=1, sweep=sweep)
        self.on_step = on_step

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        samples = self.llm_prior.sample(1, schema, verbose, False)

        chain_length = self.burn_in + n_samples * self.thinning
        keys_pool: list[str] = []
        for step in tqdm(
            range(chain_length),
            disable=pbar is None,
            position=pbar,
            desc=f"Chain {pbar}",
            dynamic_ncols=True,
        ):
            current = samples[-1].copy()
            keys = list(current.keys())
            np.random.shuffle(keys)

            if self.sweep:
                if not keys_pool:
                    keys_pool = keys
                key_to_resample = keys_pool.pop(0)
            else:
                key_to_resample = keys[-1]

            observed_without_key = {k: current[k] for k in keys if k != key_to_resample}
            conditional_schema = {
                "type": "object",
                "properties": {key_to_resample: schema["properties"][key_to_resample]},
                "required": [key_to_resample],
            }

            conditional_observed = {**observed_without_key, **(observed or {})}
            resampled_value = self.llm_prior.sample_conditional(
                1,
                conditional_schema,
                conditional_observed,
                verbose,
            )[0]
            new_sample = observed_without_key | resampled_value
            samples.append(new_sample)

            if self.on_step is not None:
                self.on_step(step, new_sample, key_to_resample)

        return samples[self.burn_in :: self.thinning][:n_samples]


def build_schema(example_ids: list[int]) -> dict[str, Any]:
    properties = {str(uid): {"type": "boolean"} for uid in example_ids}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }


def apply_assignment_to_demos(
    demonstrations: dict[int, dict[str, Any]],
    assignment: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    labelled = deepcopy(demonstrations)
    for key, value in assignment.items():
        labelled[int(key)]["label"] = int(bool(value))
    return labelled


def evaluate_assignment(
    demonstrations: dict[int, dict[str, Any]],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    labelled = apply_assignment_to_demos(demonstrations, assignment)
    return {
        "train_accuracy": float(
            np.mean([v["label"] == v["vanilla_label"] for v in labelled.values()])
        ),
        "train_predict_distribution": dict(Counter([v["label"] for v in labelled.values()])),
        "train_label_distribution": dict(Counter([v["vanilla_label"] for v in labelled.values()])),
        "train_size": len(labelled),
    }


def run_gibbs_search(
    demonstrations: dict[int, dict[str, Any]],
    whole_ids: list[int],
    args: argparse.Namespace,
    llm: OpenAICompatLLM,
    log_path: Path | str | None,
    *,
    verbose: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    schema = build_schema(whole_ids)

    sweep = bool(getattr(args, "sweep", False))
    parallel = getattr(args, "num_workers", 1) > 1
    if not parallel:
        print(
            f"[gibbs]: full-batch Gibbs over N={len(whole_ids)} "
            f"variables (T={llm.temperature}, burn_in={args.burn_in}, "
            f"thinning={args.thinning}, num_samples={args.num_samples}, sweep={sweep})"
        )

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    base_prior = CustomPrior(llm=llm, demonstrations=demonstrations)

    state = {"step": 0}

    def on_step(_local_step: int, current: dict[str, Any], resampled_key: str) -> None:
        state["step"] += 1
        if not verbose:
            return
        metrics = evaluate_assignment(demonstrations, current)
        if log_path is not None:
            log_record = {
                "step": state["step"],
                "resampled_uid": int(resampled_key),
                "label_after": bool(current[resampled_key]),
                **metrics,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_record, default=str) + "\n")

        if verbose:
            print(
                f"[step {state['step']:>5}] resampled uid={int(resampled_key):>4} "
                f"-> {bool(current[resampled_key])} | "
                f"acc={metrics['train_accuracy']:.3f} "
                f"pred_dist={metrics['train_predict_distribution']}"
            )

    gibbs = LoggedGibbsLLMPrior(
        base_prior=base_prior,
        burn_in=args.burn_in,
        thinning=args.thinning,
        sweep=sweep,
        on_step=on_step,
    )
    samples = gibbs.sample(
        n_samples=args.num_samples,
        schema=schema,
        verbose=False,
        pbar=None if parallel else True,
    )

    sample_keys = list(samples[0].keys())
    true_fractions = {
        key: sum(int(bool(s[key])) for s in samples) / len(samples) for key in sample_keys
    }
    final_assignment = {key: true_fractions[key] > 0.5 for key in sample_keys}
    final_demos = apply_assignment_to_demos(demonstrations, final_assignment)
    for key, frac in true_fractions.items():
        final_demos[int(key)]["_predicted_score"] = float(frac)

    final_metric = {
        "train_accuracy": float(
            np.mean([v["label"] == v["vanilla_label"] for v in final_demos.values()])
        ),
        "train_predict_distribution": dict(Counter(v["label"] for v in final_demos.values())),
        "train_label_distribution": dict(Counter(v["vanilla_label"] for v in final_demos.values())),
        "train_size": len(final_demos),
    }
    return final_demos, final_metric
