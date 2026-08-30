from priorbot.priors import GibbsLLMPrior

from sampling.run import (
    get_args,
    output_filename,
    save_kvar_samples,
    save_samples,
    skip_method,
)
from sampling.targets import get_target
from sampling_continuation.continuation import (
    ContinuationBarkerGibbsLLMPrior,
    ContinuationGamblingGibbsLLMPrior,
    ContinuationLLMPrior,
    ContinuationOpenAICompatLLM,
)
from sampling_continuation.templates import create_template
from sampling_continuation.utils import RESULTS_DIR


def main():
    args = get_args(__doc__)

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

    def make_llm(temperature: float) -> ContinuationOpenAICompatLLM:
        # Values and choice answers are a few tokens; the reasoning pass raises
        # its own cap inside generate_choice.
        llm = ContinuationOpenAICompatLLM(
            model_name=args.model_name,
            base_url=args.base_url,
            system_prompt=system_prompt,
            use_chat_api=args.model_type == "instruct",
            temperature=temperature,
            max_tokens=32,
        )
        llm.reasoning_max_tokens = args.max_tokens_reasoning
        return llm

    def field_prior(method: str, shuffle: bool = True) -> ContinuationLLMPrior:
        """The field-method engine. ``shuffle`` randomizes the coordinate order
        per draw, so no key is pinned to a fixed position in the array."""
        return ContinuationLLMPrior(
            llm=make_llm(args.temperature),
            template=create_template(args, method),
            shuffle_variables=shuffle,
        )

    # 1. Independent sampling (a lone scalar per fresh context)
    indep_out_path = out_dir / output_filename(args, "indep")
    if not skip_method("indep", target, args, indep_out_path):
        print("\n--- Running indep (continuation) ---")
        schema = target.object_schema(args, "indep")
        results = field_prior("indep", shuffle=False).sample_parallel(
            args.n_samples_per_chain, [schema] * args.n_chains, verbose=args.verbose, pbar=True
        )
        save_samples(
            indep_out_path, [s["sample"] for chain in results for s in chain], args.n_samples
        )

    # 2. Batch sampling (one array, filled element by element in one context)
    batch_out_path = out_dir / output_filename(args, "batch")
    if not skip_method("batch", target, args, batch_out_path):
        print("\n--- Running batch (continuation) ---")
        schema = target.object_schema(args, "batch")
        results = field_prior("batch", shuffle=False).sample_parallel(
            1, [schema] * args.n_chains, verbose=args.verbose, pbar=True
        )
        save_samples(
            batch_out_path, [v for chain in results for v in chain[0]["samples"]], args.n_samples
        )

    # 3. Gibbs sampling
    gibbs_out_path = out_dir / output_filename(args, "gibbs")
    if not skip_method("gibbs", target, args, gibbs_out_path):
        print("\n--- Running gibbs (continuation) ---")
        gibbs_prior = GibbsLLMPrior(
            llm_prior=field_prior("gibbs"),
            burn_in=args.burn_in,
            thinning=args.thinning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        samples = gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [target.object_schema(args, "gibbs")] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gibbs_out_path, samples, args.gibbs_k_vars, args.n_samples)

    # 4. Barker-Gibbs sampling
    barker_out_path = out_dir / output_filename(args, "barker_gibbs")
    if not skip_method("barker_gibbs", target, args, barker_out_path):
        print("\n--- Running Barker-Gibbs (continuation) ---")
        llm = make_llm(1.0)
        barker_gibbs_prior = ContinuationBarkerGibbsLLMPrior(
            llm=llm,
            template=create_template(args, "barker_gibbs"),
            burn_in=args.burn_in,
            thinning=args.thinning * 2,  # *2 because samples can be rejected
            manual_reasoning=args.manual_reasoning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        barker_samples = barker_gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [target.object_schema(args, "barker_gibbs")] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(barker_out_path, barker_samples, args.gibbs_k_vars, args.n_samples)

    # 5. Gambling-Gibbs sampling; deterministic unless reasoning adds randomness.
    gambling_out_path = out_dir / output_filename(args, "gambling_gibbs")
    if not skip_method("gambling_gibbs", target, args, gambling_out_path):
        print("\n--- Running Gambling-Gibbs (continuation) ---")
        llm = make_llm(1.0 if args.manual_reasoning else 0.0)
        gambling_gibbs_prior = ContinuationGamblingGibbsLLMPrior(
            llm=llm,
            template=create_template(args, "gambling_gibbs"),
            burn_in=args.burn_in,
            thinning=args.thinning * 2,  # *2 because samples can be rejected
            manual_reasoning=args.manual_reasoning,
            block_size=args.gibbs_block_size,
            sweep=args.sweep,
        )
        gambling_samples = gambling_gibbs_prior.sample_parallel(
            kvar_n_samples // args.n_chains,
            [target.object_schema(args, "gambling_gibbs")] * args.n_chains,
            verbose=args.verbose,
            pbar=True,
        )
        save_kvar_samples(gambling_out_path, gambling_samples, args.gibbs_k_vars, args.n_samples)


if __name__ == "__main__":
    main()
