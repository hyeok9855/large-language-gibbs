import argparse
import json
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerGibbsLLMPrior, GamblingGibbsLLMPrior, GibbsLLMPrior, LLMPrior

from common.utils import MODEL_NAME_TO_TYPE
from sampling.targets import TARGETS, get_target
from sampling.templates import create_template
from sampling.utils import RESULTS_DIR, indexed_var_names

# The decision kernels are the only methods with a manual-reasoning variant.
REASONING_METHODS = ("barker_gibbs", "gambling_gibbs")


def get_args(description: str | None = None) -> Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target", type=str, default="gaussian", choices=TARGETS)
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
        help="Number of scalar draws.",
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
        help="Block size for Gibbs sampling.",
    )
    parser.add_argument(
        "--gibbs_k_vars",
        type=int,
        default=16,
        help="Number of coordinates (iid copies of the target).",
    )
    parser.add_argument("--burn_in", type=int, default=None)
    parser.add_argument("--thinning", type=int, default=None)
    parser.add_argument("--no_sweep", dest="sweep", action="store_false")
    parser.add_argument(
        "--manual_reasoning",
        action="store_true",
        help=(
            "Ask for a chain of thought before the answer. Only the decision kernels "
            f"({', '.join(REASONING_METHODS)}) define this; every other method self-skips."
        ),
    )
    parser.add_argument(
        "--max_tokens_reasoning",
        type=int,
        default=4096,
        help=(
            "Output budget added to the 32-token answer for a --manual_reasoning "
            "decision-kernel call, covering the JSON reasoning field. Uniform across "
            "models so the protocol is identical; it is a cap, not a target, so a terse "
            "model pays nothing for a generous one. Olmo-3-32B-Think truncated at 1024."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["indep", "batch", "gibbs", "barker_gibbs", "gambling_gibbs"],
        default=["indep", "batch", "gibbs", "barker_gibbs", "gambling_gibbs"],
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # ------------------ Argument Validation ------------------
    # ---------------------------------------------------------

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

    if args.n_samples % args.gibbs_k_vars != 0:
        raise ValueError(
            f"n_samples ({args.n_samples}) must be divisible by gibbs_k_vars ({args.gibbs_k_vars})."
        )
    if (args.n_samples // args.gibbs_k_vars) % args.n_chains != 0:
        raise ValueError(
            f"(n_samples // gibbs_k_vars) ({args.n_samples // args.gibbs_k_vars}) must be "
            f"divisible by n_chains ({args.n_chains})."
        )
    return args


def output_filename(args: Namespace, method: str) -> str:
    """Result filename for ``method``."""
    reasoning_tag = "_reasoning" if args.manual_reasoning else ""
    gibbs_suffix = (
        f"_k{args.gibbs_k_vars}_b{args.gibbs_block_size}_nc{args.n_chains}_seed{args.seed}.json"
    )
    names = {
        "indep": f"independent_seed{args.seed}.json",
        "batch": f"batch_nc{args.n_chains}_seed{args.seed}.json",
        "gibbs": f"gibbs{gibbs_suffix}",
        "barker_gibbs": f"barkergibbs{reasoning_tag}{gibbs_suffix}",
        "gambling_gibbs": f"gamblinggibbs{reasoning_tag}{gibbs_suffix}",
    }
    return names[method]


def skip_method(method: str, target, args: Namespace, out_path: Path) -> bool:
    """Skip rules shared by both families. Runners add their own on top."""
    if method not in args.methods:
        return True

    if method in REASONING_METHODS and args.model_type != "instruct":
        print(f"Skipping {method}: requires an instruct model.")
        return True

    if args.manual_reasoning and method not in REASONING_METHODS:
        print(
            f"Skipping {method}: --manual_reasoning is defined only for the decision "
            f"kernels ({', '.join(REASONING_METHODS)})."
        )
        return True

    if out_path.exists():
        print(f"Results already exist at {out_path}, skipping...")
        return True

    return False


def save_samples(path: Path, samples: list, n_samples: int) -> None:
    samples = samples[:n_samples]
    assert len(samples) == n_samples
    with open(path, "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples to {path}")


def save_kvar_samples(path: Path, chain_results, k: int, n_samples: int) -> None:
    var_names = indexed_var_names(k)
    samples = []
    for s_chain in chain_results:
        for sample in s_chain:
            samples += [sample[key] for key in var_names]
    save_samples(path, samples, n_samples)


def main():
    args = get_args()

    target = get_target(args.target)
    kvar_n_samples = args.n_samples // args.gibbs_k_vars
    assert kvar_n_samples % args.n_chains == 0

    out_dir = (
        RESULTS_DIR
        / args.target
        / target.dir_name(args)
        / f"{args.model_name.replace('/', '--')}_temp{args.temperature}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

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
    indep_out_path = out_dir / output_filename(args, "indep")
    if not skip_method("indep", target, args, indep_out_path):
        print("\n--- Running Independent Sampling ---")
        llm = OpenAICompatLLM(**llm_common_kwargs, temperature=args.temperature, max_tokens=32)
        indep_template = create_template(args, "indep")
        indep_schema = target.object_schema(args, "indep")
        indep_prior = LLMPrior(llm=llm, template=indep_template, shuffle_variables=False)
        indep_samples = indep_prior.sample_parallel(
            args.n_samples_per_chain,
            [indep_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_samples(
            indep_out_path,
            [s["sample"] for s_chain in indep_samples for s in s_chain],
            args.n_samples,
        )

    # 2. Batch Sampling
    batch_out_path = out_dir / output_filename(args, "batch")
    if not skip_method("batch", target, args, batch_out_path):
        print("\n--- Running Batch Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=args.n_samples_per_chain * 32,
        )
        batch_template = create_template(args, "batch")
        batch_schema = target.object_schema(args, "batch")
        batch_prior = LLMPrior(llm=llm, template=batch_template, shuffle_variables=False)
        batch_results = batch_prior.sample_parallel(
            1, [batch_schema] * args.n_chains, verbose=args.verbose, pbar=True
        )

        save_samples(
            batch_out_path,
            [v for s_chain in batch_results for v in s_chain[0]["samples"]],
            args.n_samples,
        )

    # 3. Gibbs Sampling
    gibbs_out_path = out_dir / output_filename(args, "gibbs")
    if not skip_method("gibbs", target, args, gibbs_out_path):
        print("\n--- Running Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=args.temperature,
            max_tokens=args.gibbs_k_vars * 32,
        )
        gibbs_template = create_template(args, "gibbs")
        gibbs_schema = target.object_schema(args, "gibbs")
        llm_prior = LLMPrior(llm=llm, template=gibbs_template)
        gibbs_prior = GibbsLLMPrior(
            llm_prior=llm_prior,
            burn_in=args.burn_in,
            thinning=args.thinning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        gibbs_samples = gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gibbs_out_path, gibbs_samples, args.gibbs_k_vars, args.n_samples)

    # 4. Barker-Gibbs Sampling
    barker_out_path = out_dir / output_filename(args, "barker_gibbs")
    if not skip_method("barker_gibbs", target, args, barker_out_path):
        print("\n--- Running Barker-Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=1.0,
            max_tokens=32 + (args.max_tokens_reasoning if args.manual_reasoning else 0),
        )
        barker_template = create_template(args, "barker_gibbs")
        gibbs_schema = target.object_schema(args, "barker_gibbs")
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
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(barker_out_path, barker_samples, args.gibbs_k_vars, args.n_samples)

    # 5. Gambling-Gibbs Sampling
    gambling_out_path = out_dir / output_filename(args, "gambling_gibbs")
    if not skip_method("gambling_gibbs", target, args, gambling_out_path):
        print("\n--- Running Gambling-Gibbs Sampling ---")
        llm = OpenAICompatLLM(
            **llm_common_kwargs,
            temperature=0.0 if not args.manual_reasoning else 1.0,
            max_tokens=32 + (args.max_tokens_reasoning if args.manual_reasoning else 0),
        )
        gambling_template = create_template(args, "gambling_gibbs")
        gibbs_schema = target.object_schema(args, "gambling_gibbs")
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
            kvar_n_samples // args.n_chains,
            [gibbs_schema] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gambling_out_path, gambling_samples, args.gibbs_k_vars, args.n_samples)


if __name__ == "__main__":
    main()
