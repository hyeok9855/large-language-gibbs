from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from consistent_reasoning.algorithms.gibbs import (
    CustomGibbsLLMPrior,
    CustomLLMPrior,
    SweepSchedule,
    apply_assignment_to_demos,
    build_schema,
    evaluate_assignment,
)
from consistent_reasoning.models import OpenAICompatLLM


class CustomGamblingLLMPrior(CustomLLMPrior):
    def __init__(
        self,
        llm: OpenAICompatLLM,
        demonstrations: dict[int, dict[str, Any]],
        manual_reasoning: bool = False,
    ):
        super().__init__(llm, demonstrations)
        self.manual_reasoning = manual_reasoning
        if not self.llm.instruction_tuned:
            raise NotImplementedError(
                "Gambling Gibbs variant is only supported for instruction-tuned models."
            )
        self._label_choices = ["Place Bet", "Do Not Place Bet"]

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        schedule: SweepSchedule,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        key, example, demos = self.prepare_conditional(schema, observed, schedule)

        is_true_option1 = random.choice([True, False])

        opt1 = "True" if is_true_option1 else "False"
        opt2 = "False" if is_true_option1 else "True"

        bet_value = np.round(np.random.rand() * 100, 2)

        prompt = (
            f"{example['prompt']}\n"
            f"You will be presented with two options for a data point. One of these is real and "
            f"the other is fake.\nYou have the opportunity to place a bet of ${bet_value} that "
            "Option 1 is more plausible, which will pay out $100 if you are correct. Your aim is "
            f"to maximise profit.\nOption 1: {opt1}\nOption 2: {opt2}\n"
        )

        if self.manual_reasoning:
            reasoning_description = (
                "A brief explanation of the reasoning behind the decision to place a bet or not. "
                "This should be the first field in the JSON object."
            )
            gen_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": reasoning_description},
                    "bet": {"type": "string", "enum": self._label_choices},
                },
                "required": ["reasoning", "bet"],
            }
            prompt += f"Respond with JSON that conforms to this schema: {json.dumps(gen_schema)}."
            generate_max_trials = 20
        else:
            prompt += "Respond with 'Place Bet' or 'Do Not Place Bet'."
            gen_schema = self._label_choices
            generate_max_trials = 10

        chosen = self.llm.generate(
            prompt,
            schema=gen_schema,
            verbose=verbose,
            history=demos,
            max_trials=generate_max_trials,
        )

        if self.manual_reasoning:
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
            print(f"[GamblingGibbs] uid={example['uid']} -> {value} (chosen={chosen!r})")

        return [{key: value} for _ in range(n_samples)]


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

    sweep = bool(args.sweep)
    fast_demo_order = bool(getattr(args, "fast_demo_order", False))
    parallel = args.num_workers > 1
    if not parallel:
        print(
            f"[gambling_gibbs]: full-batch Gambling Gibbs over N={len(whole_ids)} "
            f"variables (T={llm.temperature}, burn_in={args.burn_in}, "
            f"thinning={args.thinning}, num_samples={args.num_samples}, sweep={sweep}, "
            f"fast_demo_order={fast_demo_order})"
        )

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    llm_prior = CustomGamblingLLMPrior(
        llm=llm,
        demonstrations=demonstrations,
        manual_reasoning=args.manual_reasoning,
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

    gibbs = CustomGibbsLLMPrior(
        llm_prior=llm_prior,
        burn_in=args.burn_in,
        thinning=args.thinning,
        sweep=sweep,
        fast_demo_order=fast_demo_order,
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
    final_assignment = {
        key: (bool(random.randint(0, 1)) if frac == 0.5 else frac > 0.5)
        for key, frac in true_fractions.items()
    }
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
