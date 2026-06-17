#!/usr/bin/env python3
"""Launch DAG-GFlowNet training jobs across multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from structure_learning.utils.llm_data_utils import get_llm_data_run_name
from structure_learning.utils.misc_utils import MODEL_NAME_TO_TYPE, STRUCTURE_LEARNING_DIR

DATASETS_DIR = STRUCTURE_LEARNING_DIR / "datasets"
TRAIN_SCRIPT = STRUCTURE_LEARNING_DIR / "dag_gflownet" / "train.py"
LOG_DIR = STRUCTURE_LEARNING_DIR / "tmp"

DATASET_PARAMS = {
    "bnrep_tubercolosis": {"burnin": 100, "thinning": 10, "block_size": 1, "sweep": True},
    "bnrep_knowledge": {"burnin": 120, "thinning": 12, "block_size": 1, "sweep": True},
    "bnrep_algalactivity2": {"burnin": 160, "thinning": 16, "block_size": 1, "sweep": True},
    "bnrep_disputed1": {"burnin": 110, "thinning": 11, "block_size": 2, "sweep": True},
    "bnrep_consequenceCovid": {"burnin": 150, "thinning": 15, "block_size": 2, "sweep": True},
}

LLM_DATA_SAMPLING_METHODS = frozenset({"direct", "gibbs", "barker_gibbs", "gambling_gibbs"})


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


def build_data_path(
    dataset_name: str,
    sampling_method: str,
    model_name: str,
    seed: int,
    manual_reasoning: bool,
) -> Path:
    params = DATASET_PARAMS[dataset_name]
    temp = 0.0 if sampling_method == "gambling_gibbs" else 1.0
    filename = get_llm_data_run_name(
        sampling_method=sampling_method,
        temperature=temp,
        top_p=1.0,
        n_samples=200,
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


def build_exp_name(
    model_name: str,
    sampling_method: str,
    gamma: float,
    seed: int,
    manual_reasoning: bool,
) -> str:
    model_slug = model_name.replace("/", "--")
    reasoning_suffix = "_reasoning" if manual_reasoning else ""
    temp = 0.0 if sampling_method == "gambling_gibbs" else 1.0
    return f"{model_slug}/{sampling_method}{reasoning_suffix}_temp{temp}_gamma{gamma}_sd{seed}"


def build_uninformative_exp_name(prior: str, seed: int, edge_beta: float) -> str:
    name = f"edge_beta{edge_beta}" if prior == "edge" else prior
    return f"Uninformative/{name}_sd{seed}"


def log_path_for(experiment: Experiment, gpu: int) -> Path:
    exp_slug = experiment.exp_name.replace("/", "__")
    return LOG_DIR / f"gpu{gpu}_{experiment.dataset_name}_{exp_slug}.log"


def iter_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    prior = args.prior
    for seed in args.seeds:
        for dataset_name in args.datasets:
            if dataset_name not in DATASET_PARAMS:
                raise ValueError(f"Unknown dataset: {dataset_name!r}")
            if prior != "llm_data":
                experiments.append(
                    Experiment(
                        dataset_name=dataset_name,
                        prior=prior,
                        llm_data_sampling_method=None,
                        llm_data_base_prior=None,
                        gamma=0.0,
                        seed=seed,
                        data_path=None,
                        exp_name=build_uninformative_exp_name(prior, seed, args.edge_beta),
                    )
                )
                continue
            for gamma in args.gammas:
                experiments.append(
                    Experiment(
                        dataset_name=dataset_name,
                        prior=prior,
                        llm_data_sampling_method=args.llm_data_sampling_method,
                        llm_data_base_prior=args.llm_data_base_prior,
                        gamma=gamma,
                        seed=seed,
                        data_path=build_data_path(
                            dataset_name=dataset_name,
                            sampling_method=args.llm_data_sampling_method,
                            model_name=args.model_name,
                            seed=seed,
                            manual_reasoning=args.manual_reasoning,
                        ),
                        exp_name=build_exp_name(
                            model_name=args.model_name,
                            sampling_method=args.llm_data_sampling_method,
                            gamma=gamma,
                            seed=seed,
                            manual_reasoning=args.manual_reasoning,
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
            f"Launched pid={process.pid} on GPU {gpu}: "
            f"{experiment.dataset_name} {experiment.label} "
            f"gamma={experiment.gamma} seed={experiment.seed} "
            f"log={log_path}",
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
            status = "ok" if return_code == 0 else f"failed (exit {return_code})"
            if return_code != 0:
                failed += 1
            print(
                f"Finished pid={job.process.pid} on GPU {job.gpu} [{status}]: "
                f"{job.experiment.dataset_name} {job.experiment.label} "
                f"gamma={job.experiment.gamma} seed={job.experiment.seed} "
                f"log={job.log_path}",
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

    experiments = iter_experiments(args)
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
        "--llm_data_sampling_method",
        choices=sorted(LLM_DATA_SAMPLING_METHODS),
        default=None,
        help="Sampling method of the LLM prior data (required when --prior llm_data).",
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
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Data seeds, e.g. 0 1 2.",
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

    if args.prior == "llm_data":
        if args.llm_data_sampling_method is None:
            raise ValueError("--llm_data_sampling_method is required when --prior llm_data.")
        if args.model_name is None:
            raise ValueError("--model_name is required when --prior llm_data.")
        if args.model_name not in MODEL_NAME_TO_TYPE:
            raise ValueError(
                f"Unknown model_name: {args.model_name!r}. "
                f"Add it to MODEL_NAME_TO_TYPE in structure_learning/utils/misc_utils.py."
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
