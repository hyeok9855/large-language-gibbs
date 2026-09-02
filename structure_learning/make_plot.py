import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.utils import MODEL_NAME_TO_TYPE
from structure_learning.utils.misc_utils import STRUCTURE_LEARNING_DIR

# (base model, instruct model)
MODEL_FAMILIES: list[tuple[str, ...]] = [
    ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("allenai/Olmo-3-1125-32B", "allenai/Olmo-3.1-32B-Instruct"),
]
# (model, method, temp, base_prior, gamma)
AlgoKey = tuple[str, str, float | None, str | None, float | None]

BASE_DIR = STRUCTURE_LEARNING_DIR / "results"

# base_prior may carry an edge beta, e.g. "edge-beta0.9", hence [\w.-]
LLM_EXP_PATTERN = re.compile(r"^(.+?)_temp(\d+\.?\d*)(?:_base([\w.-]+?))?_gamma(\d+\.?\d*)$")
EDGE_BETA_PATTERN = re.compile(r"^edge-beta(\d+\.?\d*)$")
RUN_SUFFIX_PATTERN = re.compile(r"_sd(\d+)$")


def model_families() -> list[tuple[str, tuple[str, ...]]]:
    return [(family[0], tuple(hf.replace("/", "--") for hf in family)) for family in MODEL_FAMILIES]


def families_with_results(base: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Families that have at least one known model directory under ``base``."""
    known_slugs = {
        p.name
        for p in base.iterdir()
        if p.is_dir() and p.name.replace("--", "/") in MODEL_NAME_TO_TYPE
    }
    return [
        (base_hf, family_slugs)
        for base_hf, family_slugs in model_families()
        if any(slug in known_slugs for slug in family_slugs)
    ]


def canonical_method(method: str, model_slug: str) -> str:
    if MODEL_NAME_TO_TYPE.get(model_slug.replace("--", "/")) == "instruct":
        return f"{method}_instruct"
    return method


def _plot_method_key(method: str) -> str:
    return "edge" if "edge" in method else method


METHOD_DISPLAY_COLOR = {
    "uniform": ("Uniform", "#1f77b4"),
    "direct": ("AR-Rand.", "#ffbb78"),
    "direct_instruct": ("AR-Rand.-Inst.", "#aec7e8"),
    "gibbs": ("Gibbs", "#d62728"),
    "gibbs_instruct": ("Gibbs-Inst.", "#9467bd"),
    "barker_gibbs_instruct": ("Barker-Gibbs", "#e377c2"),
    "gambling_gibbs_instruct": ("Gambl.-Gibbs", "#98df8a"),
}

TEMP_DISPLAY = [0.0, 1.0]


def _normalize_method(method: str) -> str:
    if method.endswith("_reasoning"):
        return method[: -len("_reasoning")]
    return method


def parse_experiment(model: str, name: str) -> AlgoKey:
    """Return (model, method, temp, base_prior, gamma)."""
    name = RUN_SUFFIX_PATTERN.sub("", name)

    if model == "uninformative":
        return (model, name, None, None, None)

    m = LLM_EXP_PATTERN.match(name)
    if m:
        method = _normalize_method(m.group(1))
        temp = m.group(2)
        base_prior = m.group(3)
        gamma = m.group(4)
        return (model, method, float(temp), base_prior, float(gamma))
    return (model, _normalize_method(name), None, None, None)


def load_results(base_dir: Path) -> dict[AlgoKey, list[dict]]:
    """Walk ``base_dir/<model>/<experiment>/results.json`` and group runs."""
    grouped: dict[AlgoKey, list[dict]] = defaultdict(list)
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        if (
            model_dir.name != "uninformative"
            and model_dir.name.replace("--", "/") not in MODEL_NAME_TO_TYPE
        ):
            continue
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            results_path = exp_dir / "results.json"
            if not results_path.exists():
                continue
            key = parse_experiment(model_dir.name, exp_dir.name)
            if key[2] is not None and key[2] not in TEMP_DISPLAY:
                continue

            with open(results_path) as f:
                data = json.load(f)
            data["_model"] = model_dir.name
            data["_exp_name"] = exp_dir.name
            grouped[key].append(data)
    return grouped


def group_family_results(
    grouped: dict[AlgoKey, list[dict]],
    family_slugs: frozenset[str],
    family_base: str,
    gamma: float,
) -> dict[AlgoKey, list[dict]]:
    """Merge base and instruct runs from the same model family into one plot key."""
    merged: dict[AlgoKey, list[dict]] = defaultdict(list)

    for key, runs in grouped.items():
        model, method, temp, base_prior, gamma_key = key

        if model == "uninformative":
            if _plot_method_key(method) in METHOD_DISPLAY_COLOR:
                merged[key].extend(runs)
            continue

        if model not in family_slugs:
            continue
        if gamma_key is not None and gamma_key != gamma:
            continue

        plot_method = canonical_method(method, model)
        if plot_method not in METHOD_DISPLAY_COLOR:
            continue

        plot_key = (family_base, plot_method, temp, base_prior, gamma_key)
        merged[plot_key].extend(runs)

    return dict(merged)


def _model_sort_idx(model: str) -> tuple[int, str]:
    # Uninformative firs then LLMs alphabetically.
    return (0 if model == "uninformative" else 1, model)


def _sort_key(key: AlgoKey) -> tuple[tuple[int, str], int, float, str]:
    model, method, temp, base_prior, _gamma = key
    plot_method = _plot_method_key(method)
    method_idx = (
        list(METHOD_DISPLAY_COLOR).index(plot_method)
        if plot_method in METHOD_DISPLAY_COLOR
        else len(METHOD_DISPLAY_COLOR)
    )
    return (
        _model_sort_idx(model),
        method_idx,
        temp if temp is not None else -1.0,
        base_prior if base_prior is not None else "",
    )


def _label(key: AlgoKey) -> str:
    _model, method, temp, base_prior, _gamma = key
    plot_method = _plot_method_key(method)
    if plot_method == "edge":
        m_edge = EDGE_BETA_PATTERN.match(method)
        display = f"Edge (β={m_edge.group(1)})"
    else:
        display = METHOD_DISPLAY_COLOR[plot_method][0]
    if base_prior is not None and base_prior != "uniform":
        display = f"{display}\n({base_prior})"
    if temp is not None:
        if ("gambling" in method and temp != 0.0) or ("gambling" not in method and temp != 1.0):
            display = f"{display}\nt={temp:g}"
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
    colors = [METHOD_DISPLAY_COLOR[k[1]][1] for k in keys]

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
            print(f"{k}: {metric_label}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    if save_path is not None:
        png_path = save_path.with_name(save_path.name + ".png")
        pdf_path = save_path.with_name(save_path.name + ".pdf")
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {png_path} and {pdf_path}")
    plt.close(fig)


def plot_family(
    base: Path,
    dataset_name: str,
    family_id: str,
    family_slugs: tuple[str, ...],
    gamma: float,
    results: dict[AlgoKey, list[dict]],
) -> None:
    family_base = family_id.replace("/", "--")
    available_slugs = [slug for slug in family_slugs if (base / slug).is_dir()]

    grouped = group_family_results(
        results,
        family_slugs=frozenset(family_slugs),
        family_base=family_base,
        gamma=gamma,
    )

    if not grouped:
        raise FileNotFoundError(
            f"No matching results under {base} for family={family_id!r}, gamma={gamma}"
        )

    print(
        f"Loaded algorithms (uninformative + {family_id}, slugs={available_slugs}, gamma={gamma}):"
    )
    for key in sorted(grouped.keys(), key=_sort_key):
        names = [f"{r['_model']}/{r['_exp_name']}" for r in grouped[key]]
        label = _label(key).replace("\n", " | ")
        print(f"  {label:35s}: {len(names)} run(s)  {names}")

    metrics = [
        ("expected_shd", r"$\mathbb{E}$-SHD ($\downarrow$)"),
        ("roc_auc", r"AUROC ($\uparrow$)"),
    ]

    title = f"{dataset_name.replace('bnrep_', '')}"
    plot_stem = f"boxplot_{family_base}_gamma{gamma}"
    make_boxplot(grouped, metrics, title=title, save_path=base / f"{plot_stem}")


def main(n_samples: int, dataset_name: str, gamma: float) -> None:
    base = BASE_DIR / dataset_name / f"n{n_samples}"
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    families = families_with_results(base)
    if not families:
        available = sorted(p.name for p in base.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"No known model directories found under {base}. Available: {available}"
        )

    results = load_results(base)
    for family_id, family_slugs in families:
        try:
            plot_family(
                base,
                dataset_name=dataset_name,
                family_id=family_id,
                family_slugs=family_slugs,
                gamma=gamma,
                results=results,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {dataset_name} {family_id} gamma={gamma}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.5], help="Gamma values to plot."
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Single dataset to plot.")
    args = parser.parse_args()

    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {BASE_DIR}")

    if args.dataset_name is not None:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = sorted(
            d.name for d in BASE_DIR.iterdir() if d.is_dir() and (d / f"n{args.n_samples}").is_dir()
        )

    for dataset in dataset_names:
        for gamma in args.gammas:
            try:
                main(args.n_samples, dataset, gamma)
            except FileNotFoundError as exc:
                print(f"Skipping {dataset} gamma={gamma}: {exc}")
                continue
