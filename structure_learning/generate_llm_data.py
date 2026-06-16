import json
import math
import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerGibbsLLMPrior, GamblingGibbsLLMPrior, GibbsLLMPrior, LLMPrior

from structure_learning.utils.prompt_utils import (
    build_system_prompt,
    get_dataset_description,
    get_feature_description,
)
from structure_learning.utils.misc_utils import DATASETS_DIR, MODEL_NAME_TO_TYPE, load_meta


def build_schema(meta: dict) -> dict:
    return {
        "type": "object",
        "properties": {name: feat["schema"] for name, feat in meta["features"].items()},
        "required": list(meta["features"].keys()),
    }


def main(args: Namespace) -> None:
    dataset_meta_path = DATASETS_DIR / args.dataset_name / "meta_data.json"
    llm_output_dir = DATASETS_DIR / args.dataset_name / "llm_data"

    llm_output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(dataset_meta_path)
    n_features = len(meta["features"])

    if args.thinning is None:
        args.thinning = math.ceil((n_features * 2) / args.block_size)
    if args.burn_in is None:
        args.burn_in = min(1000, 10 * args.thinning)

    schema = build_schema(meta)
    system_prompt = (
        ""
        if args.model_type == "base"
        else build_system_prompt(meta, "generating realistic data points")
    )

    llm = OpenAICompatLLM(
        base_url=args.base_url,
        model_name=args.model_name,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=4096 if args.manual_reasoning else 512,
    )
    llm._use_chat_api = True
    if args.model_type == "base":
        llm._use_chat_api = False

    match args.sampling_method:
        case "direct" | "gibbs":

            def llm_template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                variables_to_resample = [v for v in schema["required"] if v != "reasoning"]

                observed = observed or {}
                dataset_description = get_dataset_description(meta)
                feature_description = get_feature_description(
                    meta, list(observed.keys()), variables_to_resample
                )
                schema_str = json.dumps(schema)
                if observed:
                    observed_str = json.dumps(observed)
                    if args.model_type == "base":
                        return f"{dataset_description}\n{feature_description}\n[Data point] {observed_str}"
                    else:
                        required_str = '", "'.join(variables_to_resample)
                        return (
                            f"{dataset_description}\n{feature_description}\n"
                            f"We have already observed the following features: {observed_str}. "
                            f'Generate the value(s) for "{required_str}" according to the following '
                            f"schema: {schema_str}."
                        )
                else:
                    if args.model_type == "base":
                        generation_prompt = "[Data point] "
                    else:
                        generation_prompt = (
                            f"Generate a data point according to the following schema: {schema_str}"
                        )
                    return f"{dataset_description}\n{feature_description}\n{generation_prompt}"

            llm_prior = LLMPrior(
                llm=llm,
                template=llm_template,
                manual_reasoning=args.manual_reasoning,
            )

            if args.sampling_method == "direct":
                prior = llm_prior
            else:  # gibbs
                prior = GibbsLLMPrior(
                    llm_prior=llm_prior,
                    burn_in=args.burn_in,
                    thinning=args.thinning,
                    block_size=args.block_size,
                    sweep=args.sweep,
                )

        case "barker_gibbs":

            def barker_template(
                option1: dict[str, Any],
                option2: dict[str, Any],
                output_schema: dict[str, Any],  # `Option 1` or `Option 2`
                observed: dict[str, Any] | None = None,
            ) -> str:
                observed = observed or {}
                dataset_description = get_dataset_description(meta)
                feature_description = get_feature_description(
                    meta, list(observed.keys()), list(option1.keys())
                )
                template = f"{dataset_description}\n{feature_description}\n"
                if observed:
                    observed_str = json.dumps(observed)
                    template = template + (
                        f"Given the observed features with these values: {observed_str}, "
                        f"which of the following two options is more likely to be a valid "
                        "data point? "
                    )
                else:
                    template = template + (
                        "Which of the following two options is more likely to be a valid "
                        "data point? "
                    )
                option1_str = json.dumps(option1)
                option2_str = json.dumps(option2)
                output_schema_str = json.dumps(output_schema)
                return template + (
                    f"Option 1: {option1_str}. Option 2: {option2_str}. "
                    f"Respond with JSON that follows this schema: {output_schema_str}."
                )

            prior = BarkerGibbsLLMPrior(
                llm=llm,
                template=barker_template,
                burn_in=args.burn_in,
                thinning=args.thinning,
                block_size=args.block_size,
                sweep=args.sweep,
                manual_reasoning=args.manual_reasoning,
            )

        case "gambling_gibbs":

            assert args.model_type == "instruct", "Gambling prior only supports instruct LLM type"

            def gambling_template(
                option1: dict[str, Any],
                option2: dict[str, Any],
                output_schema: dict[str, Any],  # `Place Bet` or `Do Not Place Bet`
                bet_value: float,
                observed: dict[str, Any] | None = None,
            ) -> str:
                observed = observed or {}
                dataset_description = get_dataset_description(meta)
                feature_description = get_feature_description(
                    meta, list(observed.keys()), list(option1.keys())
                )
                template = f"{dataset_description}\n{feature_description}\n"
                if observed:
                    observed_str = json.dumps(observed)
                    template = template + (
                        "You will be presented with two sets of feature values for a data point, "
                        f"along with some observed features with these values: {observed_str}. "
                    )
                else:
                    template = template + (
                        "You will be presented with two sets of feature values for a data point. "
                    )
                option1_str = json.dumps(option1)
                option2_str = json.dumps(option2)
                output_schema_str = json.dumps(output_schema)
                return template + (
                    "One of the following two options is real and the other is fake. You have the "
                    f"opportunity to place a bet of ${bet_value} that Option 1 is more plausible, "
                    "which will pay out $100 if you are correct. Your aim is to maximise profit. "
                    f"Option 1 is {option1_str} and Option 2 is {option2_str}. "
                    f"Respond with JSON that follows this schema: {output_schema_str}."
                )

            prior = GamblingGibbsLLMPrior(
                llm=llm,
                template=gambling_template,
                burn_in=args.burn_in,
                thinning=args.thinning,
                block_size=args.block_size,
                sweep=args.sweep,
                manual_reasoning=args.manual_reasoning,
            )
        case _:
            raise ValueError(f"Invalid sampling method: {args.sampling_method}")

    # Sampling
    n_samples_per_chain = (args.n_samples // args.n_chains) + (
        1 if args.n_samples % args.n_chains > 0 else 0
    )
    samples_per_chain = prior.sample_parallel(
        n_samples_per_chain, [schema] * args.n_chains, verbose=args.verbose, pbar=args.pbar
    )
    samples = [sample for chain_samples in samples_per_chain for sample in chain_samples]
    samples = samples[: args.n_samples]

    columns = list(meta["features"].keys())
    df = pd.DataFrame(samples, columns=columns, dtype="category")

    # Rename back to the original column names if needed
    if args.dataset_name in ["bnrep_tubercolosis"]:
        df = df.rename(columns={"Tuberculosis": "Tubercolosis"})
    if args.dataset_name in ["bnrep_knowledge"]:
        df = df.rename(columns={"C#": "C"})

    # Saving
    run_name = f"{args.model_name.replace('/', '--')}_{args.sampling_method}_temp{args.temperature}_topp{args.top_p}"
    if args.sampling_method != "direct":
        run_name += f"_burnin{args.burn_in}_thinning{args.thinning}"

    # Only tag non-default block / sweep settings so vanilla Gibbs filenames remain
    # unchanged and existing results are not shadowed by new block/sweep runs.
    if "gibbs" in args.sampling_method:
        if args.block_size != 1:
            run_name += f"_block{args.block_size}"
    if args.manual_reasoning:
        run_name += "_reasoning"
    run_name += f"_n{args.n_samples}_sd{args.seed}"

    df.to_csv(llm_output_dir / f"{run_name}.csv", index=False)
    print(f"Saved {len(df)} samples to {llm_output_dir / run_name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate LLM prior data using priorbot.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api_key", type=str, default="NOT_A_KEY")
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=["direct", "gibbs", "barker_gibbs", "gambling_gibbs"],
        default="gibbs",
    )
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=1, help="Block size for Gibbs sampling.")
    parser.add_argument(
        "--no_sweep", dest="sweep", action="store_false", help="Disable sweep for Gibbs sampling."
    )
    parser.add_argument("--manual_reasoning", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_pbar", dest="pbar", action="store_false", default=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.base_url is None:
        if args.port is None:
            raise ValueError("Either base_url or port must be provided.")
        args.base_url = f"http://localhost:{args.port}/v1"

    if args.manual_reasoning and args.temperature == 0.0:
        warnings.warn("Manual reasoning requires temperature > 0.0, setting temperature to 1.0")
        args.temperature = 1.0

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.model_type = MODEL_NAME_TO_TYPE[args.model_name]

    os.environ["OPENAI_API_KEY"] = args.api_key

    main(args)
    print("Done!")
