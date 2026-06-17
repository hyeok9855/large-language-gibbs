def get_llm_data_run_name(
    *,
    sampling_method: str,
    temperature: float,
    top_p: float,
    n_samples: int,
    seed: int,
    burn_in: int | None = None,
    thinning: int | None = None,
    block_size: int = 1,
    sweep: bool = True,
    manual_reasoning: bool = False,
) -> str:
    run_name = f"{sampling_method}_temp{temperature}_topp{top_p}"
    if "gibbs" in sampling_method:
        if burn_in is None or thinning is None:
            raise ValueError("burn_in and thinning are required for non-direct sampling methods")
        run_name += f"_burnin{burn_in}_thinning{thinning}_block{block_size}"
        if sweep:
            run_name += "_sweep"
    if manual_reasoning:
        run_name += "_reasoning"
    return f"{run_name}_n{n_samples}_sd{seed}"
