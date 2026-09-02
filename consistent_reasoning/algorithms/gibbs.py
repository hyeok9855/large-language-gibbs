from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from priorbot.priors import GibbsLLMPrior, LLMPrior
from tqdm import tqdm

from consistent_reasoning.models import OpenAICompatLLM
from consistent_reasoning.prompt_utils import get_judge_prompt_fewshot


class SweepSchedule:
    """Pick which variable to resample next, and order the demos for each conditional.

    Flat mode (``key_to_cid=None``): a flat random scan or sweep.
    ``group_rank``/``key_to_rank`` stay ``None`` so ``order_demos`` reshuffles the
    demo order per conditional.

    Hierarchical mode (requires ``sweep=True``): resample one consistency group at a
    time, reusing one demo order per sweep. Drawing the demo order once per sweep,
    and finishing a group before starting the next group: prompts for each group
    share a prefix that vLLM's cache can reuse. Every variable is still resampled
    once per sweep and conditions on all the others, but the order is randomised per
    sweep rather than per conditional, so this approximates the default sampler.
    """

    def __init__(self, keys: list[str], sweep: bool, key_to_cid: dict[str, Any] | None = None):
        if key_to_cid is not None and not sweep:
            raise ValueError("Hierarchical scheduling (key_to_cid) requires sweep=True.")

        self.keys = list(keys)
        self.sweep = sweep

        # for flat sweep
        self._flat_queue: list[str] = []

        # for hierarchical sweep
        self._cid_to_keys: dict[Any, list[str]] | None = None
        if key_to_cid is not None:
            self._cid_to_keys = {}
            for key in self.keys:
                self._cid_to_keys.setdefault(key_to_cid[key], []).append(key)

        self.group_rank: dict[Any, int] | None = None
        self.key_to_rank: dict[int, int] | None = None
        self._cid_to_key_order: dict[Any, list[str]] = {}
        self._cid_queue: list[Any] = []
        self._member_queue: list[str] = []

    def next_key(self) -> str:
        if self._cid_to_keys is None:
            # Flat random scan: uniform draw each step.
            if not self.sweep:
                return random.choice(self.keys)
            # Flat sweep: pop from a queue reshuffled once per sweep.
            if not self._flat_queue:
                self._flat_queue = list(self.keys)
                random.shuffle(self._flat_queue)
            return self._flat_queue.pop(0)

        # Hierarchical sweep: finish one consistency group before the next,
        # redrawing the group/member order once per sweep.
        if not self._member_queue:
            if not self._cid_queue:
                cids = list(self._cid_to_keys)
                random.shuffle(cids)
                self.group_rank = {cid: rank for rank, cid in enumerate(cids)}
                self.key_to_rank = {}
                self._cid_to_key_order = {}
                for cid in cids:
                    members = list(self._cid_to_keys[cid])
                    random.shuffle(members)
                    self._cid_to_key_order[cid] = members
                    for rank, key in enumerate(members):
                        self.key_to_rank[int(key)] = rank
                self._cid_queue = cids
            self._member_queue = list(self._cid_to_key_order[self._cid_queue.pop(0)])
        return self._member_queue.pop(0)

    def order_demos(
        self, current_pool: dict[int, dict[str, Any]], target_consistency_id: Any, target_uid: int
    ) -> list[dict[str, Any]]:
        """Order the labelled pool for one conditional: other groups first, target's group last."""
        groups: dict[Any, list[dict[str, Any]]] = {}
        target_group: list[dict[str, Any]] = []

        for uid, example in current_pool.items():
            if uid == target_uid:
                continue

            if example["consistency_id"] == target_consistency_id:
                target_group.append(example)
            else:
                groups.setdefault(example["consistency_id"], []).append(example)

        group_keys = list(groups.keys())
        if self.group_rank is None:  # Flat random scan or sweep
            random.shuffle(group_keys)
        else:  # Hierarchical sweep
            group_keys.sort(key=lambda key: self.group_rank[key])

        def _ordered_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
            members = list(members)
            if self.key_to_rank is None:  # Flat random scan or sweep
                random.shuffle(members)
                return members
            # Hierarchical sweep
            return sorted(members, key=lambda e: self.key_to_rank[e["uid"]])

        ordered: list[dict[str, Any]] = []
        for group_key in group_keys:
            ordered.extend(_ordered_members(groups[group_key]))

        ordered.extend(_ordered_members(target_group))
        return ordered


class CustomLLMPrior(LLMPrior):
    def __init__(self, llm: OpenAICompatLLM, demonstrations: dict[int, dict[str, Any]]):
        super().__init__(llm)
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

    def prepare_conditional(
        self,
        schema: dict[str, Any],
        observed: dict[str, Any],
        schedule: SweepSchedule,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """Return (key, target example, demonstrations ordered for that target)."""
        if len(schema["properties"]) != 1:
            raise ValueError(
                f"{type(self).__name__}.sample_conditional expects a single-key schema; "
                f"got {list(schema['properties'])}."
            )

        (key,) = schema["properties"].keys()
        example = self.demonstrations[int(key)]

        pool: dict[int, dict[str, Any]] = {}
        for observed_key, value in observed.items():
            uid = int(observed_key)
            if uid == example["uid"]:
                continue

            labelled = deepcopy(self.demonstrations[uid])
            labelled["label"] = int(bool(value))
            pool[uid] = labelled

        demos = schedule.order_demos(
            pool,
            target_consistency_id=example["consistency_id"],
            target_uid=example["uid"],
        )
        return key, example, demos

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        schedule: SweepSchedule,
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        key, example, demos = self.prepare_conditional(schema, observed, schedule)

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
            print(f"[Gibbs] uid={example['uid']} -> {value} (chosen={chosen!r})")

        return [{key: value} for _ in range(n_samples)]

    def sample_parallel(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Not used for this experiment.")

    def sample_conditional_parallel(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Not used for this experiment.")


class CustomGibbsLLMPrior(GibbsLLMPrior):
    def __init__(
        self,
        llm_prior: CustomLLMPrior,
        burn_in: int,
        thinning: int,
        sweep: bool = False,
        fast_demo_order: bool = False,
        on_step: Callable[[int, dict[str, Any], str], None] | None = None,
    ):
        if fast_demo_order and not sweep:
            raise ValueError("fast_demo_order=True requires sweep=True.")
        super().__init__(llm_prior, burn_in, thinning, block_size=1, sweep=sweep)
        self.on_step = on_step
        self.fast_demo_order = fast_demo_order

    def _build_schedule(self, keys: list[str]) -> SweepSchedule:
        if not self.fast_demo_order:
            return SweepSchedule(keys, sweep=self.sweep)

        demonstrations = self.llm_prior.demonstrations
        key_to_cid = {key: demonstrations[int(key)]["consistency_id"] for key in keys}
        return SweepSchedule(keys, sweep=self.sweep, key_to_cid=key_to_cid)

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
        schedule = self._build_schedule(list(samples[-1].keys()))
        for step in tqdm(
            range(chain_length),
            disable=pbar is None,
            position=pbar,
            desc=f"Chain {pbar}",
            dynamic_ncols=True,
        ):
            current = samples[-1].copy()
            key_to_resample = schedule.next_key()
            observed_without_key = {k: v for k, v in current.items() if k != key_to_resample}
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
                schedule,
                verbose,
            )[0]
            new_sample = observed_without_key | resampled_value
            samples.append(new_sample)

            if self.on_step is not None:
                self.on_step(step, new_sample, key_to_resample)

        return samples[self.burn_in + self.thinning :: self.thinning][:n_samples]


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
    sweep = bool(args.sweep)
    fast_demo_order = bool(getattr(args, "fast_demo_order", False))
    parallel = args.num_workers > 1
    if not parallel:
        print(
            f"[gibbs]: full-batch Gibbs over N={len(whole_ids)} "
            f"variables (T={llm.temperature}, burn_in={args.burn_in}, "
            f"thinning={args.thinning}, num_samples={args.num_samples}, sweep={sweep}, "
            f"fast_demo_order={fast_demo_order})"
        )

    if log_path is not None:
        log_path = Path(log_path)
        log_path.unlink(missing_ok=True)

    base_prior = CustomLLMPrior(llm=llm, demonstrations=demonstrations)

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
        llm_prior=base_prior,
        burn_in=args.burn_in,
        thinning=args.thinning,
        sweep=sweep,
        fast_demo_order=fast_demo_order,
        on_step=on_step,
    )
    samples = gibbs.sample(
        n_samples=args.num_samples,
        schema=build_schema(whole_ids),
        verbose=False,
        pbar=not parallel,
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
