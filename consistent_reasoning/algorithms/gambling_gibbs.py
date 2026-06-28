from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from priorbot.priors import GibbsLLMPrior, Prior
from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.algorithms.gibbs import (
    _hierarchical_demo_order,
    build_schema,
    apply_assignment_to_demos,
    evaluate_assignment,
)


class CustomGamblingPrior(Prior):
    def __init__(
        self,
        llm: OpenAICompatLLM,
        demonstrations: dict[int, dict[str, Any]],
        manual_reasoning: bool = False,
    ):
        super().__init__()
        self.llm = llm
        self.demonstrations = demonstrations
        self.manual_reasoning = manual_reasoning
        if not self.llm.instruction_tuned:
            raise NotImplementedError(
                "Gambling Gibbs variant is only supported for instruction-tuned models."
            )
        self._label_choices = ["Place Bet", "Do Not Place Bet"]

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
                "CustomGamblingPrior.sample_conditional expects a single-key schema; "
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

        is_true_option1 = random.choice([True, False])

        opt1 = "True" if is_true_option1 else "False"
        opt2 = "False" if is_true_option1 else "True"

        bet_value = np.round(np.random.rand() * 100, 2)

        prompt = (
            f"{example['prompt']}\n"
            f"You will be presented with two options for a data point. One of these is real and the other is fake.\n"
            f"You have the opportunity to place a bet of ${bet_value} that Option 1 is more plausible, "
            f"which will pay out $100 if you are correct. Your aim is to maximise profit.\n"
            f"Option 1: {opt1}\n"
            f"Option 2: {opt2}\n"
        )

        binary_schema = {
            "type": "object",
            "properties": {"bet": {"type": "string", "enum": self._label_choices}},
            "required": ["bet"],
        }

        if getattr(self, "manual_reasoning", False):
            reasoning_description = (
                "A brief explanation of the reasoning behind the decision to place a bet or not. "
                "This should be the first field in the JSON object."
            )
            binary_schema["properties"] = {
                "reasoning": {"type": "string", "description": reasoning_description},
                **binary_schema["properties"],
            }
            binary_schema["required"] = ["reasoning"] + binary_schema["required"]
            prompt += (
                f"Respond with JSON that conforms to this schema: {json.dumps(binary_schema)}."
            )
            generate_max_trials = 20
        else:
            prompt += f"Respond with 'Place Bet' or 'Do Not Place Bet'."
            generate_max_trials = 10

        chosen = self.llm.generate(
            prompt,
            schema=(
                binary_schema if getattr(self, "manual_reasoning", False) else self._label_choices
            ),
            verbose=verbose,
            history=demos,
            max_trials=generate_max_trials,
        )

        if getattr(self, "manual_reasoning", False):
            if not isinstance(chosen, dict):
                raise TypeError(
                    f"Expected a dict from json-constrained generation; got {type(chosen)}"
                )
            chosen_opt = chosen.get("bet", "").strip()
        else:
            if not isinstance(chosen, str):
                raise TypeError(
                    f"Expected a string from choice-constrained generation; got {type(chosen)}"
                )
            chosen_opt = chosen.strip()
        if chosen_opt == "Place Bet":
            value = is_true_option1
        elif chosen_opt == "Do Not Place Bet":
            value = not is_true_option1
        else:
            raise ValueError(f"Unexpected chosen option: {chosen}")

        if verbose:
            print(f"[GamblingGibbs] uid={uid} -> {value} (chosen={chosen!r})")

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


class LoggedGamblingGibbsLLMPrior(GibbsLLMPrior):
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


def run_gambling_gibbs_search(
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
            f"[gambling_gibbs]: full-batch Gambling Gibbs over N={len(whole_ids)} "
            f"variables (T={llm.temperature}, burn_in={args.burn_in}, "
            f"thinning={args.thinning}, num_samples={args.num_samples}, sweep={sweep})"
        )

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    base_prior = CustomGamblingPrior(
        llm=llm,
        demonstrations=demonstrations,
        manual_reasoning=getattr(args, "manual_reasoning", False),
    )

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

    gibbs = LoggedGamblingGibbsLLMPrior(
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
