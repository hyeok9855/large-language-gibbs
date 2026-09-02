import json
import math
import os
import random
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerGibbsLLMPrior, GamblingGibbsLLMPrior, GibbsLLMPrior, LLMPrior

from common.utils import MODEL_NAME_TO_TYPE
from structure_learning.continuation import (
    ContinuationLLMPrior,
    ContinuationOpenAICompatLLM,
    partial_json,
    quoted_choice_regex,
)
from structure_learning.prompt_record import PromptRecorder, save_prompt_record
from structure_learning.utils.bn_utils import get_feature_renames
from structure_learning.utils.llm_data_utils import get_llm_data_run_name
from structure_learning.utils.misc_utils import DATASETS_DIR, load_meta
from structure_learning.utils.prompt_utils import (
    build_system_prompt,
    get_dataset_description,
    get_feature_description,
)


def build_schema(meta: dict) -> dict:
    return {
        "type": "object",
        "properties": {name: feat["schema"] for name, feat in meta["features"].items()},
        "required": list(meta["features"].keys()),
    }


def main(args: Namespace) -> None:
    dataset_meta_path = DATASETS_DIR / args.dataset_name / "meta_data.json"
    llm_output_dir = (
        DATASETS_DIR / args.dataset_name / "llm_data" / args.model_name.replace("/", "--")
    )

    llm_output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(dataset_meta_path)
    n_features = len(meta["features"])

    if args.thinning is None:
        args.thinning = math.ceil((n_features * 2) / args.block_size)
    if args.burn_in is None:
        args.burn_in = min(1000, 10 * args.thinning)

    filename = get_llm_data_run_name(
        sampling_method=args.sampling_method,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        seed=args.seed,
        burn_in=args.burn_in,
        thinning=args.thinning,
        block_size=args.block_size,
        sweep=args.sweep,
        manual_reasoning=args.manual_reasoning,
    )
    output_path = llm_output_dir / f"{filename}.csv"
    if output_path.exists():
        print(f"Skipping: {output_path} already exists")
        return

    full_schema = build_schema(meta)
    system_prompt = "" if args.model_type == "base" else build_system_prompt(meta)

    llm_cls = ContinuationOpenAICompatLLM if "prefill" in args.sampling_method else OpenAICompatLLM
    llm = llm_cls(
        base_url=args.base_url,
        model_name=args.model_name,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=4096 if args.manual_reasoning else 512,
        use_chat_api=args.model_type == "instruct",
    )

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
                        return (
                            f"{dataset_description}\n{feature_description}\n"
                            f"[Data point] {observed_str}"
                        )
                    else:
                        required_str = '"' + '", "'.join(variables_to_resample) + '"'
                        return (
                            f"{dataset_description}\n{feature_description}\n"
                            f"We have already observed the following features: {observed_str}. "
                            f"Generate the value(s) for {required_str} according to the following "
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

            if args.manual_reasoning:
                llm_prior.reasoning_prompt = llm_prior.reasoning_prompt.replace(
                    "step-by-step", "brief"
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

        case "direct_prefill" | "gibbs_prefill":
            # Real-prefill conditioning: the decided fields are sent as text (the
            # completion prefix for base models, an assistant prefill for
            # instruct) and only the next value is grammar-constrained. Unlike
            # schema pinning, the conditioning context is tokenised canonically
            # by the tokeniser, its layout is fixed rather than sampled, and
            # nothing depends on the backend honouring JSON property order.
            if args.manual_reasoning:
                raise ValueError("Manual reasoning is not supported for prefill samplers.")

            # Feature description and schema follow the generation order
            # (observed first, then the sampling order), exactly as the
            # JSON-grammar paths render them. key_order is fixed per sample,
            # so the user message is still identical across the per-field
            # calls of one sample.
            def llm_template(
                observed: dict[str, Any] | None = None,
                next_key: str | None = None,
                key_order: list[str] | None = None,
            ) -> str | tuple[str, str]:
                if not next_key or not key_order:
                    raise ValueError("Prefill sampling requires next_key and key_order.")
                description = (
                    f"{get_dataset_description(meta)}\n"
                    f"{get_feature_description(meta, [], key_order)}"
                )
                prefill = partial_json(observed, next_key)
                if args.model_type == "base":
                    return f"{description}\n[Data point] {prefill}"
                ordered_schema = {
                    "type": "object",
                    "properties": {k: full_schema["properties"][k] for k in key_order},
                    "required": key_order,
                }
                return (
                    f"{description}\nGenerate a data point according to the "
                    f"following schema: {json.dumps(ordered_schema)}",
                    prefill,
                )

            llm_prior = ContinuationLLMPrior(llm=llm, template=llm_template)

            if args.sampling_method == "direct_prefill":
                prior = llm_prior
            else:  # gibbs_prefill
                prior = GibbsLLMPrior(
                    llm_prior=llm_prior,
                    burn_in=args.burn_in,
                    thinning=args.thinning,
                    block_size=args.block_size,
                    sweep=args.sweep,
                )

        case "barker_gibbs":
            if args.model_type != "instruct":
                raise ValueError("Barker prior only supports instruct LLM type")

            def barker_template(
                option1: dict[str, Any],
                option2: dict[str, Any],
                output_schema: dict[str, Any],  # `Option 1` or `Option 2`
                observed: dict[str, Any] | None = None,
            ) -> str:
                observed = observed or {}
                variables_to_resample = list(option1.keys())
                dataset_description = get_dataset_description(meta)
                feature_description = get_feature_description(
                    meta, list(observed.keys()), variables_to_resample
                )
                template = f"{dataset_description}\n{feature_description}\n"
                if observed:
                    observed_str = json.dumps(observed)
                    required_str = '"' + '", "'.join(variables_to_resample) + '"'
                    template = template + (
                        f"Given the observed features with these values: {observed_str}, "
                        f"which of the following two options is more likely to be valid "
                        f"for the remaining features ({required_str})? "
                    )
                else:
                    template = template + (
                        "Which of the following two options is more likely to be a "
                        "valid data point? "
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
            if args.manual_reasoning:
                prior.reasoning_prompt = prior.reasoning_prompt.replace("step-by-step", "brief")

        case "gambling_gibbs":
            if args.model_type != "instruct":
                raise ValueError("Gambling prior only supports instruct LLM type")

            def gambling_template(
                option1: dict[str, Any],
                option2: dict[str, Any],
                output_schema: dict[str, Any],  # `Place Bet` or `Do Not Place Bet`
                bet_value: float,
                observed: dict[str, Any] | None = None,
            ) -> str:
                observed = observed or {}
                variables_to_resample = list(option1.keys())
                dataset_description = get_dataset_description(meta)
                feature_description = get_feature_description(
                    meta, list(observed.keys()), variables_to_resample
                )
                template = f"{dataset_description}\n{feature_description}\n"
                if observed:
                    observed_str = json.dumps(observed)
                    required_str = '"' + '", "'.join(variables_to_resample) + '"'
                    template = template + (
                        f"You have already observed features with these values: {observed_str}, "
                        f"and here are two options for the remaining features ({required_str}): "
                    )
                else:
                    template = template + "Here are two options for a data point: "
                option1_str = json.dumps(option1)
                option2_str = json.dumps(option2)
                output_schema_str = json.dumps(output_schema)
                return template + (
                    f"Option 1: {option1_str}. Option 2: {option2_str}. "
                    "One of the two options is real and the other is fake. You can place a "
                    f"bet of ${bet_value} that Option 1 is real, which will pay out $100 if you "
                    "are correct. Your aim is to maximise profit. Would you place a bet or not? "
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
            if args.manual_reasoning:
                prior.reasoning_prompt = prior.reasoning_prompt.replace("step-by-step", "brief")

        case _:
            raise ValueError(f"Invalid sampling method: {args.sampling_method}")

    # `--save_prompt` records the real requests; every method reaches the server
    # through `llm.generate`, so one wrapper there covers all of them.
    recorder = None
    if args.save_prompt:
        if "prefill" in args.sampling_method:

            def constraint_of(schema: Any) -> Any:
                return quoted_choice_regex(list(schema["enum"]))

        else:

            def constraint_of(schema: Any) -> Any:
                return schema

        recorder = PromptRecorder(llm, constraint_of)

    # Sampling
    n_samples_per_chain = (args.n_samples // args.n_chains) + (
        1 if args.n_samples % args.n_chains > 0 else 0
    )
    samples_per_chain = prior.sample_parallel(
        n_samples_per_chain,
        [deepcopy(full_schema) for _ in range(args.n_chains)],
        verbose=args.verbose,
        pbar=args.pbar,
    )
    samples = [sample for chain_samples in samples_per_chain for sample in chain_samples]
    samples = samples[: args.n_samples]

    columns = list(meta["features"].keys())
    df = pd.DataFrame(samples, columns=columns, dtype="category")

    # Rename the LLM-facing feature names back to the original column names
    llm_to_raw = {v: k for k, v in get_feature_renames(args.dataset_name).items()}
    if llm_to_raw:
        df = df.rename(columns=llm_to_raw)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    if recorder is not None:
        save_prompt_record(
            llm_output_dir / f"{filename}.prompt.md", args, llm, recorder, output_path.name
        )


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
        choices=[
            "direct",
            "gibbs",
            "direct_prefill",
            "gibbs_prefill",
            "barker_gibbs",
            "gambling_gibbs",
        ],
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
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        default=False,
        help="Write a Markdown record of the first and last request next to the CSV.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_pbar", dest="pbar", action="store_false", default=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.base_url is None:
        if args.port is None:
            raise ValueError("Either base_url or port must be provided.")
        args.base_url = f"http://localhost:{args.port}/v1"

    os.environ["OPENAI_API_KEY"] = args.api_key

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.model_type = MODEL_NAME_TO_TYPE[args.model_name]

    if args.manual_reasoning:
        if args.model_type != "instruct":
            raise ValueError(
                f"--manual_reasoning is only supported for instruct models; got "
                f"{args.model_type} model {args.model_name!r}."
            )
        if "prefill" in args.sampling_method:
            raise ValueError(
                f"--manual_reasoning is not supported for prefill samplers "
                f"({args.sampling_method!r})."
            )
        if args.temperature <= 0.0:
            raise ValueError("--manual_reasoning requires temperature > 0.0.")

    main(args)
    print("Done!")
