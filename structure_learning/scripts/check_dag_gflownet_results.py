#!/usr/bin/env python3
"""Report missing DAG-GFlowNet result runs by target and algorithm."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "output" / "dag_gflownet"
DEFAULT_MODELS = (
    "Llama8B",
    # "Olmo32B",
    # "Llama70B",
)
DEFAULT_RUNS = (1, 2, 3)


@dataclass(frozen=True)
class ExpectedResult:
    label: str
    relative_dir: Path


UNINFORMATIVE_ALGORITHMS = [
    ("Uniform", "uniform"),
    # ("Edge", "edge_beta0.9"),
    # ("Fair", "fair"),
]

LLM_ALGORITHMS = [
    # ("Direct t=0.7", "direct_temp0.7_gamma{gamma}"),
    ("Direct t=1.0", "direct_temp1.0_gamma{gamma}"),
    ("Direct-Instruct t=1.0", "direct_instruct_temp1.0_gamma{gamma}"),
    # ("Direct t=1.0 (reasoning)", "direct_reasoning_temp1.0_gamma{gamma}"),
    # ("Gibbs t=0.7", "gibbs_temp0.7_gamma{gamma}"),
    ("Gibbs t=1.0", "gibbs_temp1.0_gamma{gamma}"),
    # ("Gibbs t=1.0 (B=2)", "gibbs_block2_temp1.0_gamma{gamma}"),
    # ("Gibbs-Instruct t=0.7", "gibbs_instruct_temp0.7_gamma{gamma}"),
    ("Gibbs-Instruct t=1.0", "gibbs_instruct_temp1.0_gamma{gamma}"),
    # ("Gibbs-Instruct t=1.0 (reasoning)", "gibbs_instruct_reasoning_temp1.0_gamma{gamma}"),
    # ("Gibbs-Instruct t=1.0 (B=2)", "gibbs_instruct_block2_temp1.0_gamma{gamma}"),
    # ("Barker t=1.0", "barker_temp1.0_gamma{gamma}"),
    # ("Gambling t=0", "gambling_temp0.0_gamma{gamma}"),
    ("Barker-Gibbs t=1.0", "barker_gibbs_temp1.0_gamma{gamma}"),
    # ("Barker-Gibbs t=1.0 (reasoning)", "barker_gibbs_reasoning_temp1.0_gamma{gamma}"),
    # ("Barker-Gibbs t=1.0 (B=2)", "barker_gibbs_block2_temp1.0_gamma{gamma}"),
    ("Gambling-Gibbs t=0", "gambling_gibbs_temp0.0_gamma{gamma}"),
    # ("Gambling-Gibbs t=0 (reasoning)", "gambling_gibbs_reasoning_temp0.0_gamma{gamma}"),
    # ("Gambling-Gibbs t=0 (B=2)", "gambling_gibbs_block2_temp0.0_gamma{gamma}"),
]

LLM_MATRIX_ALGORITHMS = [
    # ("EdgeMatrix-Parent", "mat_parent_temp1.0_mr{mix_ratio}"),
    # ("Edge-Matrix", "mat_edge_base_temp1.0_mr{mix_ratio}"),
    # ("Edge-Matrix-Instruct", "mat_edge_instruct_temp1.0_mr{mix_ratio}"),
    # ("EdgeMatrix-Cause", "mat_cause_temp1.0_mr{mix_ratio}"),
]

LLM_ONLY_ALGORITHMS = [
    (label, template.replace("_gamma{gamma}", "")) for label, template in LLM_ALGORITHMS
]


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def expected_results(
    models: tuple[str, ...],
    gammas: tuple[str, ...],
    mix_ratios: tuple[str, ...],
    runs: tuple[int, ...],
    include_llm_only: bool,
) -> list[ExpectedResult]:
    expected: list[ExpectedResult] = []

    for label, run_base in UNINFORMATIVE_ALGORITHMS:
        for run in runs:
            expected.append(
                ExpectedResult(
                    label=label, relative_dir=Path("Uninformative") / f"{run_base}_{run}"
                )
            )

    for model in models:
        for gamma in gammas:
            for label, template in LLM_ALGORITHMS:
                run_base = template.format(gamma=gamma)
                for run in runs:
                    expected.append(
                        ExpectedResult(
                            label=f"{model} {label} gamma={gamma}",
                            relative_dir=Path(model) / f"{run_base}_{run}",
                        )
                    )

        for mix_ratio in mix_ratios:
            for label, template in LLM_MATRIX_ALGORITHMS:
                run_base = template.format(mix_ratio=mix_ratio)
                for run in runs:
                    expected.append(
                        ExpectedResult(
                            label=f"{model} {label} mix_ratio={mix_ratio}",
                            relative_dir=Path(model) / f"{run_base}_{run}",
                        )
                    )

        if include_llm_only:
            for label, run_base in LLM_ONLY_ALGORITHMS:
                for run in runs:
                    expected.append(
                        ExpectedResult(
                            label=f"{model}_only {label}",
                            relative_dir=Path(f"{model}_only") / f"{run_base}_{run}",
                        )
                    )

    return expected


def missing_results(
    target_dir: Path,
    expected: list[ExpectedResult],
    required_file: str,
) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    n100_dir = target_dir / "n100"

    for item in expected:
        run_dir = n100_dir / item.relative_dir
        complete = run_dir.is_dir() and (run_dir / required_file).is_file()
        if not complete:
            missing.setdefault(item.label, []).append(str(item.relative_dir))

    return missing


def print_missing(output_dir: Path, expected: list[ExpectedResult], required_file: str) -> int:
    targets = sorted(path for path in output_dir.iterdir() if path.is_dir())
    total_missing = 0

    for target_dir in targets:
        missing = missing_results(target_dir, expected, required_file)
        missing_count = sum(len(paths) for paths in missing.values())
        total_missing += missing_count

        if not missing:
            print(f"{target_dir.name}: OK")
            continue

        print(f"{target_dir.name}: MISSING {missing_count}")
        for label, paths in sorted(missing.items()):
            run_numbers = ", ".join(path.rsplit("_", 1)[-1] for path in paths)
            print(f"  {label}: missing runs {run_numbers}")
            for path in paths:
                print(f"    n100/{path}/{required_file}")
        print()

    if total_missing:
        print(f"Total missing: {total_missing}")
    else:
        print("All expected results are present.")

    return 1 if total_missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model dirs to check."
    )
    parser.add_argument(
        "--gammas",
        default="all",
        help="Comma-separated gammas to check, or 'all' to use all gammas discovered in the output dir.",
    )
    parser.add_argument(
        "--runs", default="1,2,3", help="Comma-separated run IDs expected for each algorithm."
    )
    parser.add_argument(
        "--required-file", default="results.json", help="File required inside each run directory."
    )
    parser.add_argument(
        "--include-llm-only",
        action="store_true",
        help="Also check model dirs named '<model>_only' with run names that do not include gamma.",
    )
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        parser.error(f"Output directory does not exist: {args.output_dir}")

    models = parse_csv(args.models)
    runs = tuple(int(run) for run in parse_csv(args.runs))

    gammas = ("0.5",)
    mix_ratios = ("0.0", "0.1", "0.2", "0.5")

    expected = expected_results(models, gammas, mix_ratios, runs, args.include_llm_only)
    print(f"Checking {len(expected)} expected result directories per target in {args.output_dir}")
    print(f"Models: {', '.join(models)}")
    print(f"Gammas: {', '.join(gammas)}")
    print(f"Mix ratios: {', '.join(mix_ratios)}")
    print(f"Runs: {', '.join(map(str, runs))}")
    print(f"Required file: {args.required_file}\n")

    return print_missing(args.output_dir, expected, args.required_file)


if __name__ == "__main__":
    raise SystemExit(main())
