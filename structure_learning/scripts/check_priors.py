from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

EXCLUDE = {
    "bnrep_bullet",
    "bnrep_wheat",
    "bnrep_algalactivity2",
    "bnrep_compaction",
    "bnrep_rockburst",
    "bnrep_disputed3",
    "bnrep_firerisk",
}

# Models and their variants
BASE_MODELS = [
    "meta-llama--Llama-3.1-8B",
    # "meta-llama--Llama-3.1-70B",
    "allenai--Olmo-3-1125-32B",
    # "google--gemma-4-31B",
]
INSTRUCT_MODELS = [
    "meta-llama--Llama-3.1-8B-Instruct",
    # "meta-llama--Llama-3.1-70B-Instruct",
    "allenai--Olmo-3-32B-Think",
    # "google--gemma-4-31B-it",
]


# Known file-naming patterns derived from bnrep_knowledge (complete case).
# seed convention: _n200_sd{seed}.csv (seeds 1,2,3)
GIBBS_TEMPS = ["1.0"]
DIRECT_TEMPS = ["1.0"]
REASONING = ["", "_reasoning"]
# REASONING = [""]


def direct_name(model, temp, reasoning):
    return f"{model}_direct_temp{temp}_topp1.0{reasoning}_n200"


def gibbs_name(model, burnin, thinning, temp, block_size, reasoning):
    block = f"_block{block_size}" if block_size > 1 else ""
    return (
        f"{model}_gibbs_temp{temp}_topp1.0_burnin{burnin}_thinning{thinning}{block}{reasoning}_n200"
    )


def barker_name(model, burnin, thinning, block_size, reasoning):
    block = f"_block{block_size}" if block_size > 1 else ""
    return (
        f"{model}_barker_temp1.0_topp1.0_burnin{burnin}_thinning{thinning}{block}{reasoning}_n200"
    )


def barker_gibbs_name(model, burnin, thinning, block_size, reasoning):
    block = f"_block{block_size}" if block_size > 1 else ""
    return f"{model}_barker_gibbs_temp1.0_topp1.0_burnin{burnin}_thinning{thinning}{block}{reasoning}_n200"


def gambling_name(model, burnin, thinning, block_size, reasoning):
    block = f"_block{block_size}" if block_size > 1 else ""
    return (
        f"{model}_gambling_temp0.0_topp1.0_burnin{burnin}_thinning{thinning}{block}{reasoning}_n200"
    )


def gambling_gibbs_name(model, burnin, thinning, block_size, reasoning):
    temp = 0.0
    if reasoning:
        temp = 1.0
    block = f"_block{block_size}" if block_size > 1 else ""
    return f"{model}_gambling_gibbs_temp{temp}_topp1.0_burnin{burnin}_thinning{thinning}{block}{reasoning}_n200"


def get_burnin_thinning(ds_name: str, block_size: int, reasoning: str):
    # Fetch burnin and thinning
    if ds_name == "bnrep_algalactivity2":
        burnin = 160
        thinning = 16
    elif ds_name == "bnrep_consequenceCovid":
        if block_size == 1:
            burnin = 300
            thinning = 30
        if block_size == 2:
            burnin = 150
            thinning = 15
        else:
            raise ValueError(f"Unknown block size: {block_size}")
    elif ds_name == "bnrep_disputed1":
        if block_size == 1:
            burnin = 220
            thinning = 22
        elif block_size == 2:
            burnin = 110
            thinning = 11
        else:
            raise ValueError(f"Unknown block size: {block_size}")
    elif ds_name == "bnrep_knowledge":
        burnin = 120
        thinning = 12
    elif ds_name == "bnrep_tubercolosis":
        burnin = 100
        thinning = 10
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    return burnin, thinning


def expected_bases(ds_name: str):
    bases = []
    block_size = 2 if ds_name in ["bnrep_consequenceCovid", "bnrep_disputed1"] else 1

    # Gibbs with base models at multiple temps
    for model in BASE_MODELS:
        for t in GIBBS_TEMPS:
            burnin, thinning = get_burnin_thinning(ds_name, block_size, "")
            bases.append(
                (
                    "gibbs_base_" + model.split("--")[1] + f"_t{t}",
                    gibbs_name(model, burnin, thinning, t, block_size, ""),
                )
            )

    # Gibbs with instruction-tuned models at multiple temps
    for model in INSTRUCT_MODELS:
        for t in GIBBS_TEMPS:
            for reasoning in REASONING:
                burnin, thinning = get_burnin_thinning(ds_name, block_size, reasoning)
                bases.append(
                    (
                        "gibbs_it_" + model.split("--")[1] + f"_t{t}",
                        gibbs_name(model, burnin, thinning, t, block_size, reasoning),
                    )
                )

    # Direct at multiple temps (only instruction-tuned models have direct based on the sample)
    for model in INSTRUCT_MODELS:
        for t in DIRECT_TEMPS:
            for reasoning in REASONING:
                burnin, thinning = get_burnin_thinning(ds_name, block_size, reasoning)
                bases.append(
                    ("direct_" + model.split("--")[1] + f"_t{t}", direct_name(model, t, reasoning))
                )

    # Barker / barker_gibbs / gambling / gambling_gibbs (only instruction-tuned models)
    for model in INSTRUCT_MODELS:
        for reasoning in REASONING:
            # bases.append(("barker_" + model.split("--")[1], barker_name(model)))
            burnin, thinning = get_burnin_thinning(ds_name, block_size, reasoning)
            bases.append(
                (
                    "barker_gibbs_" + model.split("--")[1],
                    barker_gibbs_name(model, burnin, thinning, block_size, reasoning),
                )
            )
            # bases.append(("gambling_" + model.split("--")[1], gambling_name(model)))
            bases.append(
                (
                    "gambling_gibbs_" + model.split("--")[1],
                    gambling_gibbs_name(model, burnin, thinning, block_size, reasoning),
                )
            )

    return bases


def check_dataset(ds_dir: Path):
    llm = ds_dir / "llm_data"
    if not llm.exists():
        return {"missing_dir": True, "missing": [], "extras": []}
    existing = set(p.name for p in llm.iterdir() if p.is_file())

    missing = []
    for tag, base in expected_bases(ds_dir.name):
        for seed in (1, 2, 3):
            fname = f"{base}_sd{seed}.csv"
            if fname not in existing:
                missing.append(fname)

    # extras: any files not matching expected
    expected_all = set()
    for tag, base in expected_bases(ds_dir.name):
        for seed in (1, 2, 3):
            expected_all.add(f"{base}_sd{seed}.csv")
    extras = sorted(f for f in existing if f not in expected_all)

    return {"missing_dir": False, "missing": missing, "extras": extras}


def main():
    datasets = sorted(p for p in DATASETS_DIR.iterdir() if p.is_dir() and p.name not in EXCLUDE)

    for ds in datasets:
        res = check_dataset(ds)
        if res["missing_dir"]:
            print(f"=== {ds.name} ===  NO llm_data directory")
            continue

        status = "OK" if not res["missing"] else f"MISSING {len(res['missing'])}"
        print(f"=== {ds.name} === {status}")
        if res["missing"]:
            # Group missing by base (strip _sdN.csv)
            from collections import defaultdict

            groups = defaultdict(list)
            for f in res["missing"]:
                base = f.rsplit("_sd", 1)[0]
                seed = f.rsplit("_sd", 1)[1].replace(".csv", "")
                groups[base].append(seed)
            for base, seeds in sorted(groups.items()):
                seeds_str = ",".join(sorted(seeds))
                if seeds_str == "1,2,3":
                    print(f"  MISSING ALL SEEDS: {base}")
                else:
                    print(f"  MISSING seeds {seeds_str}: {base}")
        if res["extras"]:
            print(f"  (extras not in spec: {len(res['extras'])} files)")
        print()


if __name__ == "__main__":
    main()
