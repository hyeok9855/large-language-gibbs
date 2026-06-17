import asyncio
import json
import math
import os
import random
import logging
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from consistent_reasoning.models import ModelAPI
from consistent_reasoning.prompt_utils import (
    Prompt,
    get_judge_prompt_fewshot,
    extract_claim_logprobs,
    _make_judge_prompt_creator
)
from consistent_reasoning.pipeline import Pipeline, PipelineConfig
from consistent_reasoning.dataloaders import load_assignments

logger = logging.getLogger(__name__)


def calculate_accuracy(train_data, inconsistent_pairs):
    train_probs = []
    for i in train_data.values():
        if i["label"] is None:
            continue
        if i["label"] == 1:
            train_probs.append(i["score"])
        else:
            train_probs.append(-i["score"])
    if len(train_probs) == 0:
        train_prob = 0
    else:
        train_prob = np.mean(train_probs)

    return {
        "train_accuracy": (
            0
            if len(train_data) == 0
            else np.mean([i["label"] == i["vanilla_label"] for i in train_data.values()])
        ),
        "train_label_distribution": Counter([i["vanilla_label"] for i in train_data.values()]),
        "train_predict_distribution": Counter([i["label"] for i in train_data.values()]),
        "train_prob": train_prob,
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
    }


def update_assign(data):
    for key, value in data.items():
        if value["score"] > 0:
            value["label"] = 1
        else:
            value["label"] = 0
    return data


def pick_two_inconsistent_claims(data):
    inconsistent_pairs = {}
    return inconsistent_pairs


def get_pipeline(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
    no_trailing_space=False,
    instruction_tuned=False,
    system_prompt="",
):
    pipeline_name = f"iterative-truth-assign-iter-{iter}"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    pipeline_config = PipelineConfig(
        pipeline_name,
        anthropic_num_threads=40,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step("get_assign", load_assignments, assignment)

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "choice_2",
            "consistency_id",
            "source",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append({saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i})

        for idx, key in enumerate(keys):
            tmp_keys, tmp_values = [], []
            for j, (prev_key, prev_value) in enumerate(zip(keys, values)):
                if j != idx:
                    tmp_keys.append(prev_key)
                    tmp_values.append(prev_value)

            demos = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(tmp_keys, tmp_values))
            }

            sorted_demos = {}
            for k, v in demos.items():
                q = v["consistency_id"]
                if q not in sorted_demos:
                    sorted_demos[q] = []
                sorted_demos[q].append((k, v))

            out_sorted_demos = {}
            for group in sorted_demos.values():
                for k, v in group:
                    out_sorted_demos[k] = v

            copy_data[key]["demonstration"] = out_sorted_demos

        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    get_train_preds = pipeline.add_query_step(
        "get_train_preds",
        model,
        _make_judge_prompt_creator(no_trailing_space, instruction_tuned, system_prompt),
        extract_claim_logprobs,
        dependencies=[merged_train_data],
        logprobs=20,
        max_tokens=1,
        use_cache=use_cache,
    )

    pick_claims = pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds, pick_claims],
    )
    return pipeline


async def predict_assignment(
    model_api,
    model,
    example,
    demonstrations,
    no_trailing_space=False,
    instruction_tuned=False,
    system_prompt="",
):
    demos = [v for k, v in demonstrations.items() if k != example["uid"] and v["label"] is not None]

    if instruction_tuned:
        assert not no_trailing_space
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for demo in demos:
            messages.append({"role": "user", "content": demo["prompt"]})
            messages.append({"role": "assistant", "content": "True" if demo["label"] else "False"})
        messages.append({"role": "user", "content": example["prompt"]})
        prompt = messages
    else:
        prompt = get_judge_prompt_fewshot(example, demos, pipeline=False)
        if no_trailing_space:
            from consistent_reasoning.prompt_utils import _strip_one_trailing_space
            prompt = _strip_one_trailing_space(prompt)

    anthropic_requests = [
        model_api(
            model,
            prompt,
            logprobs=20,
            max_tokens=1,
            parse_fn=extract_claim_logprobs,
        )
    ]
    responses = await asyncio.gather(*anthropic_requests)
    score = responses[0][0]["score"]
    new_label = score > 0
    return int(new_label)


def get_temperature(iteration, initial_temp, final_temp, decay_rate, schedule="exp"):
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]


def run_icm_search(
    demonstrations,
    whole_ids,
    args,
    model_api,
    log_path,
    *,
    pipeline_name,
):
    cur_metric = {
        "train_prob": -1e6,
        "inconsistent_num": 100000,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }

    print(
        "init random labels = ",
        Counter([i["label"] for i in demonstrations.values() if i["type"] == "seed"]),
        "init label acc = ",
        np.mean(
            [
                i["label"] == i["vanilla_label"]
                for i in demonstrations.values()
                if i["type"] == "seed"
            ]
        ),
    )

    no_trailing_space = bool(getattr(args, "no_trailing_space", False))
    instruction_tuned = bool(getattr(args, "instruction_tuned", False))
    system_prompt = getattr(args, "system_prompt", "")

    iter = 0
    flip_cnt = 0
    example_id = 0

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}
        if iter == 0:
            pipeline = get_pipeline(
                args.model,
                name=pipeline_name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
                no_trailing_space=no_trailing_space,
                instruction_tuned=instruction_tuned,
                system_prompt=system_prompt,
            )
            results = asyncio.run(pipeline.run())
            cur_metric = results["evaluate"]

        cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}

        while True:
            candidates_ids = whole_ids
            weights = [1 for _ in range(len(candidates_ids))]
            for i in candidates_ids:
                if i in cur_pool:
                    same_consistency_group_ids = [
                        j
                        for j in candidates_ids
                        if demonstrations[j]["consistency_id"]
                        == demonstrations[i]["consistency_id"]
                    ]
                    for j in same_consistency_group_ids:
                        if j not in cur_pool:
                            weights[j] = 100

            example_id = random.choices(candidates_ids, k=1, weights=weights)[0]
            break

        new_label = asyncio.run(
            predict_assignment(
                model_api,
                args.model,
                demonstrations[example_id],
                cur_pool,
                no_trailing_space=no_trailing_space,
                instruction_tuned=instruction_tuned,
                system_prompt=system_prompt,
            )
        )

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label

            tmp_pool = {k: v for k, v in tmp_demonstrations.items() if v["label"] is not None}
            pipeline = get_pipeline(
                model=args.model,
                name=pipeline_name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
                no_trailing_space=no_trailing_space,
                instruction_tuned=instruction_tuned,
                system_prompt=system_prompt,
            )
            results = asyncio.run(pipeline.run())
            metric = results["evaluate"]
            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )
            print(
                f"iter = {iter}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']}, new acc = {metric['train_accuracy']}, cur score = {get_energy(cur_metric, args.alpha)}, new score = {get_energy(metric, args.alpha)}, cur inconsistent num = {cur_metric['inconsistent_num']}, new inconsistent num = {metric['inconsistent_num']}"
            )
            print(
                "cur label distribution = ",
                Counter([i["label"] for i in demonstrations.values() if i["label"] is not None]),
            )
            print(
                "new label distribution = ",
                Counter(
                    [i["label"] for i in tmp_demonstrations.values() if i["label"] is not None]
                ),
            )

            accept_prob = math.exp(
                (get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)) / T
            )
            print("accept prob = ", accept_prob)
            if random.random() < accept_prob:
                print("accept")
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "iter": iter,
                                    "flip_cnt": flip_cnt,
                                    "acc": cur_metric["train_accuracy"],
                                    "score": get_energy(cur_metric, args.alpha),
                                }
                            )
                            + "\n"
                        )
            else:
                print("reject")

        print("=" * 100)
        iter += 1

    return demonstrations, cur_metric
