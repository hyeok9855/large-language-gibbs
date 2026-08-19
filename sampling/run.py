import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerGibbsLLMPrior, GamblingGibbsLLMPrior, GibbsLLMPrior, LLMPrior

from sampling.continuation_llm import ContinuationOpenAICompatLLM
from sampling.continuation_prior import ContinuationLLMPrior
from sampling.targets import TARGETS, get_target
from sampling.templates import create_continuation_setup, create_template_and_schema
from sampling.utils import MODEL_NAME_TO_TYPE, RESULTS_DIR, indexed_var_names


def _skip_method(method: str, target, args: argparse.Namespace, out_path: Path) -> bool:
    if method not in args.methods:
        return True

    if target.is_joint and method in ("indep", "batch"):
        print(f"Skipping {method}: not defined for joint target {target.name}.")
        return True

    if "gibbs" in method and target.name == "multinomial" and args.gibbs_block_size < 2:
        print(
            f"Skipping {method}: multinomial needs --gibbs_block_size >= 2 "
            "(single-site updates are degenerate)."
        )
        return True

    if "continuation" in method and args.manual_reasoning:
        print(f"Skipping {method}: --manual_reasoning is not supported for continuation methods.")
        return True

    if method in ("barker_gibbs", "gambling_gibbs") and args.model_type != "instruct":
        print(f"Skipping {method}: requires an instruct model.")
        return True

    if out_path.exists():
        print(f"Results already exist at {out_path}, skipping...")
        return True

    return False


def main(args: argparse.Namespace):

    target = get_target(args.target)
    kvar_n_samples = args.n_samples if target.is_joint else args.n_samples // args.gibbs_k_vars

    out_dir = (
        RESULTS_DIR
        / args.target
        / target.dir_name(args)
        / f"{args.model_name.replace('/', '--')}_temp{args.temperature}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reasoning_tag = "_reasoning" if args.manual_reasoning else ""
    kvar_suffix = (
        f"_k{args.gibbs_k_vars}_b{args.gibbs_block_size}_nc{args.n_chains}_seed{args.seed}.json"
    )

    system_prompt = (
        ""
        if args.model_type == "base"
        else "You are a helpful assistant that can sample from probability distributions."
    )

    llm_common_kwargs = {
        "model_name": args.model_name,
        "base_url": args.base_url,
        "system_prompt": system_prompt,
        "use_chat_api": args.model_type == "instruct",
    }

    # 1. Independent Sampling
    indep_out_path = out_dir / f"independent{reasoning_tag}_seed{args.seed}.json"
    if not _skip_method("indep", target, args, indep_out_path):
        print("\n--- Running Independent Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=32 + (1024 if args.manual_reasoning else 0),
        )
        indep_template, indep_schema = create_template_and_schema("indep", args)
        indep_prior = LLMPrior(
            llm=llm,
            template=indep_template,
            shuffle_variables=False,
            manual_reasoning=args.manual_reasoning,
        )
        indep_prior.reasoning_prompt = indep_prior.reasoning_prompt.replace("step-by-step", "brief")
        indep_samples = indep_prior.sample_parallel(
            args.n_samples_per_chain,
            [indep_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        indep_samples = [s["sample"] for s_chain in indep_samples for s in s_chain]
        with open(indep_out_path, "w") as f:
            json.dump(indep_samples, f)
        print(f"Saved {len(indep_samples)} samples to {indep_out_path}")

    # 2. Batch Sampling
    batch_out_path = out_dir / f"batch{reasoning_tag}_nc{args.n_chains}_seed{args.seed}.json"
    if not _skip_method("batch", target, args, batch_out_path):
        print("\n--- Running Batch Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=args.n_samples_per_chain * 32 + (1024 if args.manual_reasoning else 0),
        )
        batch_template, batch_schema = create_template_and_schema("batch", args)
        batch_prior = LLMPrior(
            llm=llm,
            template=batch_template,
            shuffle_variables=False,
            manual_reasoning=args.manual_reasoning,
        )
        batch_prior.reasoning_prompt = batch_prior.reasoning_prompt.replace("step-by-step", "brief")
        batch_results = batch_prior.sample_parallel(
            1, [batch_schema] * args.n_chains, verbose=args.verbose, pbar=True
        )

        batch_samples_flat = []
        for s_chain in batch_results:
            batch_samples_flat += s_chain[0]["samples"]
        batch_samples_flat = batch_samples_flat[: args.n_samples]
        assert len(batch_samples_flat) == args.n_samples

        with open(batch_out_path, "w") as f:
            json.dump(batch_samples_flat, f)
        print(f"Saved {len(batch_samples_flat)} samples to {batch_out_path}")

    # --- Helper functions for saving samples ---

    def save_kvar_samples(path, chain_results):
        var_names = indexed_var_names(args.gibbs_k_vars)

        if target.is_joint:
            samples = []
            for s_chain in chain_results:
                for sample in s_chain:
                    samples.append({key: sample[key] for key in var_names})
        else:
            samples = []
            for s_chain in chain_results:
                for sample in s_chain:
                    samples += [sample[key] for key in var_names]
        samples = samples[: args.n_samples]
        assert len(samples) == args.n_samples
        with open(path, "w") as f:
            json.dump(samples, f)
        print(f"Saved {len(samples)} samples to {path}")

    def run_one_pass(out_path, *, shuffle_variables, continuation):
        assert kvar_n_samples % args.n_chains == 0
        if continuation:
            llm = ContinuationOpenAICompatLLM(
                **llm_common_kwargs,
                temperature=args.temperature,
                max_tokens=32,
            )
            template, schema = create_continuation_setup(args)
            prior = ContinuationLLMPrior(
                llm=llm,
                template=template,
                shuffle_variables=shuffle_variables,
            )
        else:
            llm = OpenAICompatLLM(
                **llm_common_kwargs,
                temperature=args.temperature,
                max_tokens=args.gibbs_k_vars * 32 + (1024 if args.manual_reasoning else 0),
            )
            template, schema = create_template_and_schema("direct", args)
            prior = LLMPrior(
                llm=llm,
                template=template,
                shuffle_variables=shuffle_variables,
                manual_reasoning=args.manual_reasoning,
            )
            prior.reasoning_prompt = prior.reasoning_prompt.replace("step-by-step", "brief")

        samples = prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [deepcopy(schema) for _ in range(args.n_chains)],
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(out_path, samples)

    # 3a. Direct Sampling (single-pass, shuffled coordinate order)
    direct_out_path = (
        out_dir
        / f"direct{reasoning_tag}_k{args.gibbs_k_vars}_nc{args.n_chains}_seed{args.seed}.json"
    )
    if not _skip_method("direct", target, args, direct_out_path):
        print("\n--- Running Direct Sampling ---")
        run_one_pass(direct_out_path, shuffle_variables=True, continuation=False)

    # 3b. Direct-fixed (single-pass, ancestral / schema order X1..Xn)
    direct_fix_out_path = (
        out_dir
        / f"direct_fixed{reasoning_tag}_k{args.gibbs_k_vars}_nc{args.n_chains}_seed{args.seed}.json"
    )
    if not _skip_method("direct_fixed", target, args, direct_fix_out_path):
        print("\n--- Running Direct-fixed Sampling ---")
        run_one_pass(direct_fix_out_path, shuffle_variables=False, continuation=False)

    # 3c. Direct continuation (field-by-field, shuffled coordinate order)
    direct_conti_out_path = (
        out_dir / f"direct_continuation_k{args.gibbs_k_vars}_nc{args.n_chains}_seed{args.seed}.json"
    )
    if not _skip_method("direct_continuation", target, args, direct_conti_out_path):
        print("\n--- Running Direct Continuation Sampling ---")
        run_one_pass(direct_conti_out_path, shuffle_variables=True, continuation=True)

    # 3d. Direct-fixed continuation (field-by-field, ancestral order)
    direct_fixconti_out_path = (
        out_dir
        / f"direct_fixed_continuation_k{args.gibbs_k_vars}_nc{args.n_chains}_seed{args.seed}.json"
    )
    if not _skip_method("direct_fixed_continuation", target, args, direct_fixconti_out_path):
        print("\n--- Running Direct-fixed Continuation Sampling ---")
        run_one_pass(direct_fixconti_out_path, shuffle_variables=False, continuation=True)

    # 4. Gibbs Sampling
    gibbs_out_path = out_dir / f"gibbs{reasoning_tag}{kvar_suffix}"
    if not _skip_method("gibbs", target, args, gibbs_out_path):
        print("\n--- Running Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=args.gibbs_k_vars * 32 + (1024 if args.manual_reasoning else 0),
        )

        gibbs_template, gibbs_schema = create_template_and_schema("gibbs", args)
        llm_prior = LLMPrior(
            llm=llm,
            template=gibbs_template,
            manual_reasoning=args.manual_reasoning,
        )
        llm_prior.reasoning_prompt = llm_prior.reasoning_prompt.replace("step-by-step", "brief")
        gibbs_prior = GibbsLLMPrior(
            llm_prior=llm_prior,
            burn_in=args.burn_in,
            thinning=args.thinning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        assert kvar_n_samples % args.n_chains == 0
        gibbs_samples = gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gibbs_out_path, gibbs_samples)

    # 4b. Gibbs with JSON continuation (field-by-field; base via completions, instruct via prefill)
    gibbs_conti_out_path = out_dir / f"gibbs_continuation{kvar_suffix}"
    if not _skip_method("gibbs_continuation", target, args, gibbs_conti_out_path):
        print("\n--- Running Gibbs Continuation Sampling ---")
        llm = ContinuationOpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=32,
        )
        continuation_template, gibbs_schema = create_continuation_setup(args)
        continuation_prior = ContinuationLLMPrior(
            llm=llm,
            template=continuation_template,
            shuffle_variables=True,
        )
        gibbs_prior = GibbsLLMPrior(
            llm_prior=continuation_prior,
            burn_in=args.burn_in,
            thinning=args.thinning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        assert kvar_n_samples % args.n_chains == 0
        gibbs_continuation_samples = gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gibbs_conti_out_path, gibbs_continuation_samples)

    # 5. Barker-Gibbs Sampling
    barker_out_path = out_dir / f"barkergibbs{reasoning_tag}{kvar_suffix}"
    if not _skip_method("barker_gibbs", target, args, barker_out_path):
        print("\n--- Running Barker-Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=1.0,
            max_tokens=32 + (1024 if args.manual_reasoning else 0),
        )
        barker_template, gibbs_schema = create_template_and_schema("barker_gibbs", args)
        barker_gibbs_prior = BarkerGibbsLLMPrior(
            llm=llm,
            template=barker_template,
            burn_in=args.burn_in,
            thinning=args.thinning * 2,  # *2 because samples can be rejected
            manual_reasoning=args.manual_reasoning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        barker_gibbs_prior.reasoning_prompt = barker_gibbs_prior.reasoning_prompt.replace(
            "step-by-step", "brief"
        )
        assert kvar_n_samples % args.n_chains == 0
        barker_samples = barker_gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(barker_out_path, barker_samples)

    # 6. Gambling-Gibbs Sampling
    gambling_out_path = out_dir / f"gamblinggibbs{reasoning_tag}{kvar_suffix}"
    if not _skip_method("gambling_gibbs", target, args, gambling_out_path):
        print("\n--- Running Gambling-Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=0.0 if not args.manual_reasoning else 1.0,
            max_tokens=32 + (1024 if args.manual_reasoning else 0),
        )
        gambling_template, gibbs_schema = create_template_and_schema("gambling_gibbs", args)
        gambling_gibbs_prior = GamblingGibbsLLMPrior(
            llm=llm,
            burn_in=args.burn_in,
            thinning=args.thinning * 2,  # *2 because samples can be rejected
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
            manual_reasoning=args.manual_reasoning,
            template=gambling_template,
        )
        gambling_gibbs_prior.reasoning_prompt = gambling_gibbs_prior.reasoning_prompt.replace(
            "step-by-step", "brief"
        )
        assert kvar_n_samples % args.n_chains == 0
        gambling_samples = gambling_gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gambling_out_path, gambling_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from probability distributions using LLMs."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="gaussian",
        choices=sorted(TARGETS),  # discrete: uniform, binomial, poisson, multinomial
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--base_url", type=str, default=None, help="Base URL for OpenAI compatible API."
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api_key", type=str, default="NOT_A_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=256,
        help=(
            "Number of scalar draws for univariate targets, or number of joint vectors "
            "for random_walk and multinomial."
        ),
    )

    # Target distribution parameters; each target contributes its own.
    for _target in TARGETS.values():
        _target.add_arguments(parser)

    # LLM parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument(
        "--gibbs_block_size",
        type=int,
        default=4,
        help="Block size for Gibbs sampling. The multinomial target requires >= 2.",
    )
    parser.add_argument(
        "--gibbs_k_vars",
        type=int,
        default=16,
        help=(
            "Number of coordinates. For univariate targets these are iid copies; "
            "for joint targets this is the dimension (8 in the default random walk "
            "and multinomial setups)."
        ),
    )
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--no_sweep", dest="sweep", action="store_false")
    parser.add_argument("--manual_reasoning", action="store_true")

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "indep",
            "batch",
            "direct",
            "direct_fixed",
            "direct_continuation",
            "direct_fixed_continuation",
            "gibbs",
            "gibbs_continuation",
            "barker_gibbs",
            "gambling_gibbs",
        ],
        default=[
            "indep",
            "batch",
            "direct",
            "direct_fixed",
            "direct_continuation",
            "direct_fixed_continuation",
            "gibbs",
            "gibbs_continuation",
            "barker_gibbs",
            "gambling_gibbs",
        ],
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.base_url is None:
        if args.port is None:
            raise ValueError("Either base_url or port must be provided.")
        args.base_url = f"http://localhost:{args.port}/v1"

    os.environ["OPENAI_API_KEY"] = args.api_key

    target = get_target(args.target)
    target.validate(args)

    if args.thinning is None:
        args.thinning = (args.gibbs_k_vars // args.gibbs_block_size) * 2
    if args.burn_in is None:
        args.burn_in = min(100, (args.n_samples // args.n_chains) * args.thinning // 10)

    args.model_type = MODEL_NAME_TO_TYPE[args.model_name]

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.n_samples % args.n_chains != 0:
        raise ValueError(
            f"n_samples ({args.n_samples}) must be divisible by n_chains ({args.n_chains})."
        )
    args.n_samples_per_chain = args.n_samples // args.n_chains

    if not target.is_joint:
        if args.n_samples % args.gibbs_k_vars != 0:
            raise ValueError(
                f"n_samples ({args.n_samples}) must be divisible by "
                f"gibbs_k_vars ({args.gibbs_k_vars})."
            )
        if (args.n_samples // args.gibbs_k_vars) % args.n_chains != 0:
            raise ValueError(
                f"(n_samples // gibbs_k_vars) ({args.n_samples // args.gibbs_k_vars}) must be "
                f"divisible by n_chains ({args.n_chains})."
            )

    main(args)
