import sys
import os

# Add root/sampling to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import random

from priorbot.llm import OpenAICompatLLM
from priorbot.priors import (
    BarkerGibbsLLMPrior,
    GamblingGibbsLLMPrior,
    GibbsLLMPrior,
    LLMPrior,
)
from utils import MODEL_NAME_TO_TYPE, RESULTS_DIR


def main(args: argparse.Namespace):

    if args.target == "uniform":
        from templates.uniform import create_template_and_schema
    elif args.target == "gaussian":
        from templates.gaussian import create_template_and_schema
    else:
        raise ValueError(f"Invalid target: {args.target}")

    def get_out_dir(args):
        outdir = RESULTS_DIR / args.target

        if args.target == "uniform":
            outdir = outdir / f"max{args.maxnum}"
        elif args.target == "gaussian":
            outdir = outdir / f"mean{args.mean}_std{args.std}"
        else:
            raise ValueError(f"Invalid target: {args.target}")

        outdir = outdir / f"{args.model_name.replace('/', '--')}_temp{args.temperature}"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    out_dir = get_out_dir(args)

    system_prompt = (
        ""
        if args.model_type == "base"
        else "You are a helpful assistant that can sample from probability distributions."
    )

    # 1. Independent Sampling
    if "indep" in args.methods:
        print("\n--- Running Independent Sampling ---")
        # Check if the results already exist
        indep_out_path = (
            out_dir
            / f"independent{'_reasoning' if args.manual_reasoning else ''}_seed{args.seed}.json"
        )
        if indep_out_path.exists():
            print(f"Results already exist at {indep_out_path}, skipping...")
        else:
            llm = OpenAICompatLLM(
                model_name=args.model_name,
                base_url=args.base_url,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=20 + (1024 if args.manual_reasoning else 0),
            )
            indep_template, indep_schema = create_template_and_schema("indep", args)
            indep_prior = LLMPrior(
                llm=llm,
                template=indep_template,
                shuffle_variables=False,
                manual_reasoning=args.manual_reasoning,
            )
            indep_prior.reasoning_prompt = indep_prior.reasoning_prompt.replace(
                "step-by-step", "brief"
            )
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
    if "batch" in args.methods:
        print("\n--- Running Batch Sampling ---")
        # Check if the results already exist
        batch_out_path = (
            out_dir
            / f"batch{'_reasoning' if args.manual_reasoning else ''}_nc{args.n_chains}_seed{args.seed}.json"
        )
        if batch_out_path.exists():
            print(f"Results already exist at {batch_out_path}, skipping...")
        else:
            llm = OpenAICompatLLM(
                model_name=args.model_name,
                base_url=args.base_url,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.n_samples_per_chain * 15 + (1024 if args.manual_reasoning else 0),
            )
            batch_template, batch_schema = create_template_and_schema("batch", args)
            batch_prior = LLMPrior(
                llm=llm,
                template=batch_template,
                shuffle_variables=False,
                manual_reasoning=args.manual_reasoning,
            )
            batch_prior.reasoning_prompt = batch_prior.reasoning_prompt.replace(
                "step-by-step", "brief"
            )
            batch_results = batch_prior.sample_parallel(
                1, [batch_schema] * args.n_chains, verbose=args.verbose, pbar=True
            )

            batch_samples_flat = []
            for s_chain in batch_results:
                batch_samples_flat += s_chain[0]["samples"]

            with open(batch_out_path, "w") as f:
                json.dump(batch_samples_flat, f)
            print(f"Saved {len(batch_samples_flat)} samples to {batch_out_path}")

    # 3. Gibbs Sampling
    if "gibbs" in args.methods:
        print("\n--- Running Gibbs Sampling ---")
        # Check if the results already exist
        gibbs_out_path = (
            out_dir
            / f"gibbs{'_reasoning' if args.manual_reasoning else ''}_k{args.gibbs_k_vars}_b{args.gibbs_block_size}_nc{args.n_chains}_seed{args.seed}.json"
        )
        if gibbs_out_path.exists():
            print(f"Results already exist at {gibbs_out_path}, skipping...")
        else:
            gibbs_n_samples = args.n_samples // args.gibbs_k_vars
            llm = OpenAICompatLLM(
                model_name=args.model_name,
                base_url=args.base_url,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.gibbs_k_vars * 20 + (1024 if args.manual_reasoning else 0),
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
            gibbs_samples = gibbs_prior.sample_parallel(
                gibbs_n_samples // args.n_chains,
                [gibbs_schema] * args.n_chains,
                verbose=args.verbose,
                pbar=True,
            )

            # flatten the structure
            gibbs_samples_flat = []
            for s_chain in gibbs_samples:
                for s in s_chain:
                    gibbs_samples_flat += [s[f"X{i}"] for i in range(args.gibbs_k_vars)]

            with open(gibbs_out_path, "w") as f:
                json.dump(gibbs_samples_flat, f)
            print(f"Saved {len(gibbs_samples_flat)} samples to {gibbs_out_path}")

    # 4. Barker-Gibbs Sampling
    if "barker" in args.methods:
        print("\n--- Running Barker-Gibbs Sampling ---")
        barker_out_path = (
            out_dir
            / f"barkergibbs{'_reasoning' if args.manual_reasoning else ''}_k{args.gibbs_k_vars}_b{args.gibbs_block_size}_nc{args.n_chains}_seed{args.seed}.json"
        )
        if barker_out_path.exists():
            print(f"Results already exist at {barker_out_path}, skipping...")
        elif args.model_type != "instruct":
            # Base models can't reliably follow the JSON-choice schema used by the acceptance step.
            print("Barker-Gibbs requires an instruct model, skipping...")
        else:
            gibbs_n_samples = args.n_samples // args.gibbs_k_vars
            llm = OpenAICompatLLM(
                model_name=args.model_name,
                base_url=args.base_url,
                system_prompt=system_prompt,
                temperature=1.0,
                max_tokens=20 + (1024 if args.manual_reasoning else 0),
            )
            barker_template, gibbs_schema = create_template_and_schema("barker", args)
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
            barker_samples = barker_gibbs_prior.sample_parallel(
                gibbs_n_samples // args.n_chains,
                [gibbs_schema] * args.n_chains,
                verbose=args.verbose,
                pbar=True,
            )

            barker_samples_flat = []
            for s_chain in barker_samples:
                for s in s_chain:
                    barker_samples_flat += [s[f"X{i}"] for i in range(args.gibbs_k_vars)]

            with open(barker_out_path, "w") as f:
                json.dump(barker_samples_flat, f)
            print(f"Saved {len(barker_samples_flat)} samples to {barker_out_path}")

    # 5. Gambling-Gibbs Sampling
    if "gambling" in args.methods:
        print("\n--- Running Gambling-Gibbs Sampling ---")
        gambling_out_path = (
            out_dir
            / f"gamblinggibbs{'_reasoning' if args.manual_reasoning else ''}_k{args.gibbs_k_vars}_b{args.gibbs_block_size}_nc{args.n_chains}_seed{args.seed}.json"
        )
        if gambling_out_path.exists():
            print(f"Results already exist at {gambling_out_path}, skipping...")
        elif args.model_type != "instruct":
            print("Gambling-Gibbs requires an instruct model, skipping...")
        else:
            gibbs_n_samples = args.n_samples // args.gibbs_k_vars
            llm = OpenAICompatLLM(
                model_name=args.model_name,
                base_url=args.base_url,
                system_prompt=system_prompt,
                temperature=0.0 if not args.manual_reasoning else 1.0,
                max_tokens=20 + (1024 if args.manual_reasoning else 0),
            )
            gambling_template, gibbs_schema = create_template_and_schema("gambling", args)
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
            gambling_samples = gambling_gibbs_prior.sample_parallel(
                gibbs_n_samples // args.n_chains,
                [gibbs_schema] * args.n_chains,
                verbose=args.verbose,
                pbar=True,
            )

            gambling_samples_flat = []
            for s_chain in gambling_samples:
                for s in s_chain:
                    gambling_samples_flat += [s[f"X{i}"] for i in range(args.gibbs_k_vars)]

            with open(gambling_out_path, "w") as f:
                json.dump(gambling_samples_flat, f)
            print(f"Saved {len(gambling_samples_flat)} samples to {gambling_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample from probability distributions using LLMs."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="gaussian",
        choices=["gaussian", "uniform"],  # uniform is discrete
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--base_url", type=str, default=None, help="Base URL for OpenAI compatible API."
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api_key", type=str, default="NOT_A_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=256)

    # Gaussian distribution parameters
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument(
        "--mcmc_sigma_multiplier",
        type=float,
        default=4.0,
        help="Multiplier of the standard deviation of the Gaussian distribution to bound the sampling space.",
    )

    # Uniform distribution parameters
    parser.add_argument("--maxnum", type=int, default=99)

    # LLM parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_chains", type=int, default=1)
    parser.add_argument(
        "--gibbs_block_size",
        type=int,
        default=4,
        help="Block size for Gibbs sampling.",
    )
    parser.add_argument(
        "--gibbs_k_vars", type=int, default=16, help="Number of variables for Gibbs sampling."
    )
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--no_sweep", dest="sweep", action="store_false")
    parser.add_argument("--manual_reasoning", action="store_true")

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["indep", "batch", "gibbs", "barker", "gambling"],
        default=["indep", "batch", "gibbs", "barker", "gambling"],
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.base_url is None:
        assert args.port is not None, "Either base_url or port must be provided."
        args.base_url = f"http://localhost:{args.port}/v1"

    os.environ["OPENAI_API_KEY"] = args.api_key

    if args.thinning is None:
        args.thinning = (args.gibbs_k_vars // args.gibbs_block_size) * 2
    if args.burn_in is None:
        args.burn_in = min(100, (args.n_samples // args.n_chains) * args.thinning // 10)

    args.model_type = MODEL_NAME_TO_TYPE.get(args.model_name, "base")

    random.seed(args.seed)

    assert args.n_samples % args.n_chains == 0
    args.n_samples_per_chain = args.n_samples // args.n_chains

    main(args)
