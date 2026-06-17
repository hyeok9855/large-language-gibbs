raise NotImplementedError("This script is not implemented yet")

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from structure_learning.utils.misc_utils import STRUCTURE_LEARNING_DIR

# (model, method, temp, gamma) — temp and gamma are None for uninformative priors
AlgoKey = tuple[str, str, float | None, float | None]

BASE_DIR = STRUCTURE_LEARNING_DIR / "results"

UNINFORMATIVE_DIR = "Uninformative"

BLOCK2_DATASETS = frozenset({"bnrep_disputed1", "bnrep_consequenceCovid"})
GIBBS_BASE_METHODS = frozenset({"gibbs", "gibbs_instruct", "barker_gibbs", "gambling_gibbs"})

LLM_EXP_PATTERN = re.compile(r"^(.+?)_temp(\d+\.?\d*)_gamma(\d+\.?\d*)$")
MAT_EDGE_EXP_PATTERN = re.compile(r"^(.+?)_temp(\d+\.?\d*)_mr(\d+\.?\d*)$")
EDGE_BETA_PATTERN = re.compile(r"^edge_beta(\d+\.?\d*)$")
RUN_SUFFIX_PATTERN = re.compile(r"_(\d+)$")

METHOD_DISPLAY = {
    "uniform": "Uniform",
    "direct": "Direct",
    "direct_instruct": "Direct-Inst.",
    "gibbs": "Gibbs",
    "gibbs_block2": "Gibbs",
    # "gibbs_block4": "Gibbs (B=4)",
    "gibbs_instruct": "Gibbs-Inst.",
    "gibbs_instruct_block2": "Gibbs-Inst.",
    # "gibbs_instruct_block4": "Gibbs-Inst. (B=4)",
    "barker_gibbs": "Barker-Gibbs",
    "barker_gibbs_block2": "Barker-Gibbs",
    # "barker_gibbs_block4": "Barker-Gibbs (B=4)",
    "gambling_gibbs": "Gambl.-Gibbs",
    "gambling_gibbs_block2": "Gambl.-Gibbs",
    # "gambling_gibbs_block4": "Gambl.-Gibbs (B=4)",
    # "barker": "Barker",
    # "gambling": "Gambling",
    # "edge": "Edge",
    # "fair": "Fair",
    # "mat_edge_base_mr0.0": "Edge-Matrix-Mix0.0",
    # "mat_edge_base_mr0.5": "Edge-Matrix-Mix0.5",
    # "mat_edge_instruct_mr0.0": "Edge-Matrix-Instruct-Mix0.0",
    # "mat_edge_instruct_mr0.5": "Edge-Matrix-Instruct-Mix0.5",
}

TEMP_DISPLAY = [0.0, 1.0]

METHOD_ORDER = [
    "uniform",
    # "mat_edge_base_mr0.0",
    # "mat_edge_base_mr0.5",
    # "mat_edge_instruct_mr0.0",
    # "mat_edge_instruct_mr0.5",
    # "edge",
    # "fair",
    "direct",
    "direct_instruct",
    "gibbs",
    "gibbs_block2",
    "gibbs_instruct",
    "gibbs_instruct_block2",
    # "barker",
    # "gambling",
    "barker_gibbs",
    "barker_gibbs_block2",
    "gambling_gibbs",
    "gambling_gibbs_block2",
]

PALETTE = {
    "uniform": "#1f77b4",
    # "mat_edge": "#aec7e8",
    # "edge": "#aec7e8",
    # "fair": "#6baed6",
    "direct": "#ffbb78",
    "direct_instruct": "#aec7e8",
    "gibbs": "#d62728",
    "gibbs_block2": "#d62728",
    "gibbs_instruct": "#9467bd",
    "gibbs_instruct_block2": "#9467bd",
    # "barker": "#ff7f0e",
    # "gambling": "#2ca02c",
    "barker_gibbs": "#e377c2",
    "barker_gibbs_block2": "#e377c2",
    "gambling_gibbs": "#98df8a",
    "gambling_gibbs_block2": "#98df8a",
}


def parse_experiment(model: str, name: str) -> AlgoKey:
    """Return (model, method, temp, gamma).

    ``temp`` and ``gamma`` are ``None`` for uninformative priors.
    """
    name = RUN_SUFFIX_PATTERN.sub("", name)

    if model == UNINFORMATIVE_DIR:
        if name == "uniform":
            return (model, "uniform", None, None)
        if name == "fair":
            return (model, "fair", None, None)
        if EDGE_BETA_PATTERN.match(name) or name == "edge":
            return (model, "edge", None, None)
        return (model, name, None, None)

    if "mat_" in name:
        m = MAT_EDGE_EXP_PATTERN.match(name)
        if m:
            return (model, m.group(1) + "_mr" + m.group(3), float(m.group(2)), None)
        raise

    m = LLM_EXP_PATTERN.match(name)
    if m:
        return (model, m.group(1), float(m.group(2)), float(m.group(3)))
    return (model, name, None, None)


def load_results(base_dir: Path) -> dict[AlgoKey, list[dict]]:
    """Walk ``base_dir/<model>/<experiment>/results.json`` and group runs."""
    grouped: dict[AlgoKey, list[dict]] = defaultdict(list)
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            results_path = exp_dir / "results.json"
            if not results_path.exists():
                continue
            key = parse_experiment(model_dir.name, exp_dir.name)
            if key[1] not in METHOD_DISPLAY:
                continue
            if key[2] is not None and key[2] not in TEMP_DISPLAY:
                continue

            with open(results_path) as f:
                data = json.load(f)
            data["_model"] = model_dir.name
            data["_exp_name"] = exp_dir.name
            grouped[key].append(data)
    return grouped


def _method_allowed(method: str, dataset_name: str) -> bool:
    use_block2 = dataset_name in BLOCK2_DATASETS
    if method.endswith("_block2"):
        return use_block2
    if method in GIBBS_BASE_METHODS:
        return not use_block2
    return True


def _model_sort_idx(model: str) -> tuple[int, str]:
    # Uninformative first, then LLMs alphabetically.
    return (0 if model == UNINFORMATIVE_DIR else 1, model)


def _sort_key(key: AlgoKey) -> tuple[tuple[int, str], int, float, float]:
    model, method, temp, gamma = key
    method_idx = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (
        _model_sort_idx(model),
        method_idx,
        temp if temp is not None else -1.0,
        gamma if gamma is not None else -1.0,
    )


def _label(key: AlgoKey) -> str:
    _model, method, temp, _gamma = key
    display = METHOD_DISPLAY.get(method, method.capitalize())
    if temp is None:
        return display
    # return f"{display}\nt={temp:g}"
    return display


def make_boxplot(
    grouped: dict[AlgoKey, list[dict]],
    metrics: list[tuple[str, str]],
    title: str = "",
    figsize: tuple[float, float] | None = None,
    save_path: Path | None = None,
) -> None:
    n_metrics = len(metrics)
    keys = sorted(grouped.keys(), key=_sort_key)
    n_groups = len(keys)
    labels = [_label(k) for k in keys]
    colors = [PALETTE.get(k[1], "#999999") for k in keys]

    if figsize is None:
        figsize = (max(10.0, 1.0 * n_groups), 4.5)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        data_per_group = []
        for k in keys:
            vals = [r[metric_key] for r in grouped[k] if metric_key in r]
            data_per_group.append(vals)

        positions = np.arange(n_groups)

        multi = [i for i, d in enumerate(data_per_group) if len(d) > 1]
        single = [i for i, d in enumerate(data_per_group) if len(d) == 1]

        if multi:
            bp = ax.boxplot(
                [data_per_group[i] for i in multi],
                positions=[positions[i] for i in multi],
                widths=0.5,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="x", markersize=5, markeredgecolor="black"),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
            )
            for patch, idx in zip(bp["boxes"], multi):
                patch.set_facecolor(colors[idx])
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
                patch.set_alpha(0.85)

        for idx in single:
            ax.plot(
                positions[idx],
                data_per_group[idx][0],
                marker="D",
                markersize=6,
                color=colors[idx],
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=16, rotation=45, ha="right")
        ax.set_title(metric_label, fontsize=20, pad=8)
        ax.tick_params(axis="y", labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=24)

    fig.tight_layout()

    # print mean and std of each group
    for k in keys:
        for metric_key, metric_label in metrics:
            vals = [r[metric_key] for r in grouped[k] if metric_key in r]
            print(f"{k}: {metric_label}: {np.mean(vals):.3f}\\std{{{np.std(vals):.3f}}}")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")


def main(args):
    base = BASE_DIR / args.algorithm / args.dataset_name / f"n{args.n_samples}"
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    model_dir = base / args.model
    if not model_dir.is_dir():
        available = sorted(
            p.name for p in base.iterdir() if p.is_dir() and p.name != UNINFORMATIVE_DIR
        )
        raise FileNotFoundError(
            f"LLM model directory not found: {model_dir}. Available: {available}"
        )

    grouped = load_results(base)

    # Keep: (a) uninformative baselines, (b) the selected LLM at the requested gamma.
    grouped = {
        key: runs
        for key, runs in grouped.items()
        if _method_allowed(key[1], args.dataset_name)
        and (
            key[0] == UNINFORMATIVE_DIR
            or (key[0] == args.model and (key[3] is None or key[3] == args.gamma))
        )
    }

    print(f"Loaded algorithms ({UNINFORMATIVE_DIR} + {args.model}, gamma={args.gamma}):")
    for key in sorted(grouped.keys(), key=_sort_key):
        names = [f"{r['_model']}/{r['_exp_name']}" for r in grouped[key]]
        label = _label(key).replace("\n", " | ")
        print(f"  {label:35s}: {len(names)} run(s)  {names}")

    metrics = [
        ("expected_shd", r"$\mathbb{E}$-SHD ($\downarrow$)"),
        ("roc_auc", r"AUROC ($\uparrow$)"),
    ]

    title = f"{args.dataset_name.replace('bnrep_', '')}"
    # title = f"{args.dataset_name} | n={args.n_samples} | {args.model} | γ={args.gamma}"

    save_path = base / f"boxplot_results_{args.model}_gamma{args.gamma}.png"
    make_boxplot(grouped, metrics, title=title, save_path=save_path)
    save_path = base / f"boxplot_results_{args.model}_gamma{args.gamma}.pdf"
    make_boxplot(grouped, metrics, title=title, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="dag_gflownet")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    gammas = [
        0.1,
        # 0.2,
        # 0.5,
        # 1.0,
    ]
    models = [
        "Llama8B",
        # "Llama70B",
        # "Olmo32B",
    ]

    alg_dir = BASE_DIR / args.algorithm
    if not alg_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {alg_dir}")

    dataset_names = [d.name for d in alg_dir.iterdir() if d.is_dir()]
    for dataset in dataset_names:
        args.dataset_name = dataset
        for model in models:
            args.model = model
            for gamma in gammas:
                args.gamma = gamma
                try:
                    main(args)
                except FileNotFoundError:
                    print(f"Skipping {model} {gamma} because results not found")
                    continue
