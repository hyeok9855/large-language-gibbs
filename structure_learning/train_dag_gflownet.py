#!/usr/bin/env python3
"""Launch DAG-GFlowNet training jobs across multiple GPUs."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from common.utils import MODEL_NAME_TO_TYPE
from structure_learning.utils.llm_data_utils import get_llm_data_run_name
from structure_learning.utils.misc_utils import STRUCTURE_LEARNING_DIR, load_meta

DATASETS_DIR = STRUCTURE_LEARNING_DIR / "datasets"
TRAIN_SCRIPT = STRUCTURE_LEARNING_DIR / "dag_gflownet" / "train.py"
LOG_DIR = STRUCTURE_LEARNING_DIR / "tmp"
RESULTS_DIR = STRUCTURE_LEARNING_DIR / "results"

DATASET_PARAMS = {
    "bnrep_tubercolosis": {"burnin": 100, "thinning": 10, "block_size": 1, "sweep": True},
    "bnrep_knowledge": {"burnin": 120, "thinning": 12, "block_size": 1, "sweep": True},
    "bnrep_algalactivity2": {"burnin": 160, "thinning": 16, "block_size": 1, "sweep": True},
    "bnrep_gonorrhoeae": {"burnin": 100, "thinning": 10, "block_size": 2, "sweep": True},
    "bnrep_disputed1": {"burnin": 110, "thinning": 11, "block_size": 2, "sweep": True},
    "bnrep_cardiovascular": {"burnin": 130, "thinning": 13, "block_size": 2, "sweep": True},
    "bnrep_consequenceCovid": {"burnin": 150, "thinning": 15, "block_size": 2, "sweep": True},
}


def resolve_dataset_params(
    dataset_name: str,
    block_size: int | None = None,
    no_sweep: bool = False,
) -> dict:
    """DATASET_PARAMS entry, with the Gibbs chain rescaled for a block-size
    override the same way generate_llm_data.py derives it (thinning keeps two
    sweeps between retained samples, burn_in stays 10 thinnings, capped)."""
    params = dict(DATASET_PARAMS[dataset_name])
    if block_size is not None and block_size != params["block_size"]:
        meta = load_meta(DATASETS_DIR / dataset_name / "meta_data.json")
        thinning = math.ceil((len(meta["features"]) * 2) / block_size)
        params.update(block_size=block_size, thinning=thinning, burnin=min(1000, 10 * thinning))
    if no_sweep:
        params["sweep"] = False
    return params


@dataclass(frozen=True)
class Experiment:
    dataset_name: str
    prior: str
    llm_data_sampling_method: str | None
    llm_data_base_prior: str | None
    gamma: float
    seed: int
    data_path: Path | None
    exp_name: str

    @property
    def label(self) -> str:
        if self.prior == "llm_data":
            return f"llm_data:{self.llm_data_sampling_method}"
        return self.prior


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("At least one GPU index is required.")
    return gpus


def llm_data_temperature(sampling_method: str, manual_reasoning: bool) -> float:
    if sampling_method == "gambling_gibbs" and not manual_reasoning:
        return 0.0
    return 1.0


def build_data_path(
    dataset_name: str,
    sampling_method: str,
    model_name: str,
    seed: int,
    manual_reasoning: bool,
    block_size: int | None = None,
    no_sweep: bool = False,
) -> Path:
    params = resolve_dataset_params(dataset_name, block_size=block_size, no_sweep=no_sweep)
    temp = llm_data_temperature(sampling_method, manual_reasoning)
    filename = get_llm_data_run_name(
        sampling_method=sampling_method,
        temperature=temp,
        top_p=1.0,
        n_samples=200,
        n_chains=10,
        seed=seed,
        burn_in=params["burnin"],
        thinning=params["thinning"],
        block_size=params["block_size"],
        sweep=params["sweep"],
        manual_reasoning=manual_reasoning,
    )
    return (
        DATASETS_DIR / dataset_name / "llm_data" / model_name.replace("/", "--") / f"{filename}.csv"
    )


def build_llm_exp_name(
    model_name: str,
    sampling_method: str,
    gamma: float,
    seed: int,
    manual_reasoning: bool,
    base_prior_name: str,
    block_size: int | None = None,
    no_sweep: bool = False,
) -> str:
    model_slug = model_name.replace("/", "--")
    reasoning_suffix = "_reasoning" if manual_reasoning else ""
    temp = llm_data_temperature(sampling_method, manual_reasoning)
    # A non-default chain layout is a different arm of the experiment, so it gets
    # its own results directory; the default stays untagged to keep the paths of
    # the results already on disk. Same for the Gibbs block/sweep ablations.
    variant_suffix = ""
    if "gibbs" in sampling_method:
        if block_size is not None:
            variant_suffix += f"_block{block_size}"
        if no_sweep:
            variant_suffix += "_nosweep"
    return (
        f"{model_slug}/{sampling_method}{reasoning_suffix}"
        f"_temp{temp}_base{base_prior_name}_gamma{gamma}{variant_suffix}_sd{seed}"
    )


def log_path_for(experiment: Experiment, gpu: int) -> Path:
    exp_slug = experiment.exp_name.replace("/", "__")
    return LOG_DIR / f"gpu{gpu}_{experiment.dataset_name}_{exp_slug}.log"


def result_path_for(experiment: Experiment, num_samples: int) -> Path:
    return (
        RESULTS_DIR
        / experiment.dataset_name
        / f"n{num_samples}"
        / experiment.exp_name
        / "results.json"
    )


def iter_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    prior = args.prior
    if prior != "llm_data":
        name = prior
        if name == "edge":
            name += f"-beta{args.edge_beta}"

        for seed in args.seeds:
            for dataset_name in args.datasets:
                experiments.append(
                    Experiment(
                        dataset_name=dataset_name,
                        prior=prior,
                        llm_data_sampling_method=None,
                        llm_data_base_prior=None,
                        gamma=0.0,
                        seed=seed,
                        data_path=None,
                        exp_name=f"uninformative/{name}_sd{seed}",
                    )
                )

    else:
        base_prior_name = args.llm_data_base_prior
        if base_prior_name == "edge":
            base_prior_name += f"-beta{args.edge_beta}"

        for gamma in args.gammas:
            for seed in args.seeds:
                for dataset_name in args.datasets:
                    for sampling_method in args.llm_data_sampling_methods:
                        if args.manual_reasoning and "continuation" in sampling_method:
                            continue
                        experiments.append(
                            Experiment(
                                dataset_name=dataset_name,
                                prior=prior,
                                llm_data_sampling_method=sampling_method,
                                llm_data_base_prior=args.llm_data_base_prior,
                                gamma=gamma,
                                seed=seed,
                                data_path=build_data_path(
                                    dataset_name=dataset_name,
                                    sampling_method=sampling_method,
                                    model_name=args.model_name,
                                    seed=seed,
                                    manual_reasoning=args.manual_reasoning,
                                    block_size=args.llm_block_size,
                                    no_sweep=args.llm_no_sweep,
                                ),
                                exp_name=build_llm_exp_name(
                                    model_name=args.model_name,
                                    sampling_method=sampling_method,
                                    gamma=gamma,
                                    seed=seed,
                                    manual_reasoning=args.manual_reasoning,
                                    base_prior_name=base_prior_name,
                                    block_size=args.llm_block_size,
                                    no_sweep=args.llm_no_sweep,
                                ),
                            )
                        )
    return experiments


def build_train_command(exp: Experiment, args: argparse.Namespace) -> list[str]:
    if exp.prior == "llm_data":
        base_prior = exp.llm_data_base_prior
        base_prior_kwargs = {"beta": args.edge_beta} if base_prior == "edge" else {}
        prior_kwargs = {
            "data_path": str(exp.data_path),
            "gamma": exp.gamma,
            "base_prior": base_prior,
            "base_prior_kwargs": base_prior_kwargs,
        }
    else:
        prior_kwargs = {"beta": args.edge_beta} if exp.prior == "edge" else {}
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--prior",
        exp.prior,
        "--prior_kwargs",
        json.dumps(prior_kwargs),
        "--exp_name",
        exp.exp_name,
        "--seed",
        str(exp.seed),
        "bn",
        "--dataset_name",
        exp.dataset_name,
        "--num_samples",
        str(args.num_samples),
        "--data_seed",
        str(args.data_seed),
    ]


@dataclass
class RunningJob:
    gpu: int
    experiment: Experiment
    process: subprocess.Popen
    log_path: Path
    log_file: TextIO


class GpuJobPool:
    def __init__(self, gpus: list[int], jobs_per_gpu: int, xla_mem_fraction: float) -> None:
        self.gpus = gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.xla_mem_fraction = xla_mem_fraction
        self.running: list[RunningJob] = []
        self._terminated = False

    def running_on_gpu(self, gpu: int) -> int:
        return sum(1 for job in self.running if job.gpu == gpu and job.process.poll() is None)

    def _close_log(self, job: RunningJob) -> None:
        if not job.log_file.closed:
            job.log_file.close()

    def launch(self, gpu: int, command: list[str], experiment: Experiment) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.xla_mem_fraction)
        log_path = log_path_for(experiment, gpu)
        log_file = log_path.open("w")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self.running.append(
            RunningJob(
                gpu=gpu,
                experiment=experiment,
                process=process,
                log_path=log_path,
                log_file=log_file,
            )
        )
        print(
            f"[LAUNCHED] pid={process.pid} on GPU {gpu}: "
            f"{experiment.dataset_name} {experiment.label} "
            f"gamma={experiment.gamma} seed={experiment.seed} "
            f"\nlog file: {log_path}",
            flush=True,
        )

    def reap_finished(self) -> int:
        failed = 0
        still_running: list[RunningJob] = []
        for job in self.running:
            return_code = job.process.poll()
            if return_code is None:
                still_running.append(job)
                continue
            self._close_log(job)
            if return_code != 0:
                failed += 1
            print(
                f"[{'FINISHED' if return_code == 0 else 'FAILED'}] "
                f"pid={job.process.pid} on GPU {job.gpu}: "
                f"{job.experiment.dataset_name} {job.experiment.label} "
                f"gamma={job.experiment.gamma} seed={job.experiment.seed} "
                f"\nlog file: {job.log_path}",
                flush=True,
            )
        self.running = still_running
        return failed

    def terminate_all(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        for job in self.running:
            if job.process.poll() is None:
                job.process.terminate()
        for job in self.running:
            if job.process.poll() is None:
                try:
                    job.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    job.process.kill()
            self._close_log(job)
        self.running.clear()

    def run(self, experiments: list[Experiment], commands: list[list[str]]) -> int:
        pending = list(zip(experiments, commands, strict=True))
        failed = 0

        def handle_signal(signum: int, _frame: object) -> None:
            print(f"\nReceived signal {signum}, terminating jobs...", flush=True)
            self.terminate_all()
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while pending or self.running:
            failed += self.reap_finished()

            for gpu in self.gpus:
                while self.running_on_gpu(gpu) < self.jobs_per_gpu and pending:
                    experiment, command = pending.pop(0)
                    self.launch(gpu, command, experiment)

            if pending or self.running:
                time.sleep(1)

        failed += self.reap_finished()
        return failed


def main(args: argparse.Namespace) -> None:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_experiments = iter_experiments(args)
    experiments = [
        exp for exp in all_experiments if not result_path_for(exp, args.num_samples).is_file()
    ]
    skipped = len(all_experiments) - len(experiments)
    if skipped:
        print(f"Skipping {skipped} experiment(s) with existing results.")

    missing = [
        exp for exp in experiments if exp.data_path is not None and not exp.data_path.is_file()
    ]
    if missing:
        missing_paths = "\n".join(f"  {exp.data_path}" for exp in missing)
        raise FileNotFoundError(
            f"{len(missing)} experiment(s) reference LLM data files that do not exist "
            f"(generate them first, or check DATASET_PARAMS):\n{missing_paths}"
        )

    commands = [build_train_command(exp, args) for exp in experiments]
    print(f"Planned {len(experiments)} experiment(s) on GPUs {args.gpus}.")

    if args.dry_run:
        for index, (exp, command) in enumerate(zip(experiments, commands, strict=True)):
            gpu = args.gpus[index % len(args.gpus)]
            log_path = log_path_for(exp, gpu)
            print(
                f"[dry-run] GPU {gpu}: {exp.exp_name}\n"
                f"  data_path={exp.data_path or '(n/a)'}\n"
                f"  log_path={log_path}\n"
                f"  command={' '.join(command)}"
            )
        return

    pool = GpuJobPool(
        gpus=args.gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        xla_mem_fraction=args.xla_mem_fraction,
    )
    failed = pool.run(experiments, commands)
    if failed:
        print(f"{failed} job(s) failed.", flush=True)
        sys.exit(1)
    print("All jobs completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch DAG-GFlowNet training experiments across multiple GPUs."
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        required=True,
        help='Comma-separated GPU indices, e.g. "0,1,2".',
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=1,
        help="Maximum number of concurrent jobs per GPU (default: %(default)s).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names, e.g. bnrep_tubercolosis bnrep_knowledge.",
    )
    parser.add_argument(
        "--prior",
        choices=["uniform", "edge", "fair", "llm_data"],
        required=True,
        help="Prior type passed to train.py: an uninformative prior (uniform, edge, fair) "
        "or llm_data.",
    )
    parser.add_argument(
        "--llm_data_sampling_methods",
        nargs="+",
        default=None,
        choices=["direct", "gibbs", "barker_gibbs", "gambling_gibbs"],
        help="Sampling methods of the LLM prior data (required when --prior llm_data).",
    )
    parser.add_argument(
        "--llm_data_base_prior",
        choices=["uniform", "edge", "fair"],
        default="uniform",
        help="Uninformative base prior mixed into the llm_data prior (default: %(default)s).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model name (required when --prior llm_data).",
    )
    parser.add_argument(
        "--manual_reasoning",
        action="store_true",
        help="Use prior data generated with manual reasoning.",
    )
    parser.add_argument(
        "--llm_block_size",
        type=int,
        default=None,
        help=(
            "Gibbs block size the LLM prior data was generated with, when it differs "
            "from the DATASET_PARAMS default. Rescales burn-in/thinning the same way "
            "generate_llm_data.py does and tags the results directory with _block<B>."
        ),
    )
    parser.add_argument(
        "--llm_no_sweep",
        action="store_true",
        help=(
            "Use LLM prior data generated with --no_sweep (random-block Gibbs). "
            "Tags the results directory with _nosweep."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Random seeds for train.py, e.g. 0 1 2; the data is fixed by --data_seed.",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.5],
        help="Gamma values for the LLM data prior (default: %(default)s).",
    )
    parser.add_argument(
        "--edge_beta",
        type=float,
        default=0.9,
        help="Beta for the edge prior (default: %(default)s).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of training samples passed to train.py (default: %(default)s).",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Random seed for data generation in train.py (default: %(default)s).",
    )
    parser.add_argument(
        "--xla_mem_fraction",
        type=float,
        default=0.1,
        help="Value for XLA_PYTHON_CLIENT_MEM_FRACTION (default: %(default)s).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned jobs without launching them.",
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_PARAMS:
            raise ValueError(f"Unknown dataset name: {dataset_name!r}")

    if args.prior == "llm_data":
        if args.llm_data_sampling_methods is None:
            raise ValueError("--llm_data_sampling_methods is required when --prior llm_data.")
        if args.model_name is None:
            raise ValueError("--model_name is required when --prior llm_data.")
        if args.model_name not in MODEL_NAME_TO_TYPE:
            raise ValueError(
                f"Unknown model_name: {args.model_name!r}. "
                f"Add it to MODEL_NAME_TO_TYPE in common/utils.py."
            )
        if args.manual_reasoning and MODEL_NAME_TO_TYPE[args.model_name] != "instruct":
            raise ValueError(
                f"Manual reasoning is only supported for instruct models; "
                f"got {MODEL_NAME_TO_TYPE[args.model_name]!r} model {args.model_name!r}."
            )

    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        sys.exit(130)
