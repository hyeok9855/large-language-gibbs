import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from structure_learning.utils.misc_utils import MODEL_NAME_TO_TYPE, STRUCTURE_LEARNING_DIR

# (model, method, temp, gamma) — temp and gamma are None for uninformative priors
AlgoKey = tuple[str, str, float | None, float | None]

BASE_DIR = STRUCTURE_LEARNING_DIR / "results"

LLM_EXP_PATTERN = re.compile(r"^(.+?)_temp(\d+\.?\d*)_gamma(\d+\.?\d*)$")
EDGE_BETA_PATTERN = re.compile(r"^edge_beta(\d+\.?\d*)$")
RUN_SUFFIX_PATTERN = re.compile(r"_sd(\d+)$")


def hf_to_slug(hf: str) -> str:
    """
    Example: "meta-llama/Llama-3.1-8B" -> "meta-llama--Llama-3.1-8B".
    """
    return hf.replace("/", "--")


def slug_to_hf(slug: str) -> str | None:
    """
    Example: "meta-llama--Llama-3.1-8B" -> "meta-llama/Llama-3.1-8B".
    """
    for hf in MODEL_NAME_TO_TYPE:
        if hf_to_slug(hf) == slug:
            return hf
    return None


def model_families() -> list[tuple[str, tuple[str, ...]]]:
    """All base/instruct families defined in MODEL_NAME_TO_TYPE."""
    bases = [hf for hf, kind in MODEL_NAME_TO_TYPE.items() if kind == "base"]
    instructs = [hf for hf, kind in MODEL_NAME_TO_TYPE.items() if kind == "instruct"]
    return [
        (base_hf, (hf_to_slug(base_hf), hf_to_slug(instruct_hf)))
        for base_hf, instruct_hf in zip(bases, instructs, strict=True)
    ]


def families_with_results(base: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Families that have at least one known model directory under ``base``."""
    known_slugs = {
        p.name
        for p in base.iterdir()
        if p.is_dir() and p.name != "uninformative" and slug_to_hf(p.name) is not None
    }
    return [
        (base_hf, family_slugs)
        for base_hf, family_slugs in model_families()
        if any(slug in known_slugs for slug in family_slugs)
    ]


def to_instruct_method(method: str) -> str:
    if method in {"direct", "gibbs"}:
        return f"{method}_instruct"
    return method


def canonical_method(method: str, model_slug: str) -> str:
    hf = slug_to_hf(model_slug)
    if hf is not None and MODEL_NAME_TO_TYPE[hf] == "instruct":
        method = to_instruct_method(method)
    return method


METHOD_DISPLAY = {
    "uniform": "Uniform",
    "direct": "Direct",
    "direct_instruct": "Direct-Inst.",
    "gibbs": "Gibbs",
    "gibbs_instruct": "Gibbs-Inst.",
    "barker_gibbs": "Barker-Gibbs",
    "gambling_gibbs": "Gambl.-Gibbs",
}

TEMP_DISPLAY = [0.0, 1.0]

METHOD_ORDER = [
    "uniform",
    "direct",
    "direct_instruct",
    "gibbs",
    "gibbs_instruct",
    "barker_gibbs",
    "gambling_gibbs",
]

PALETTE = {
    "uniform": "#1f77b4",
    "direct": "#ffbb78",
    "direct_instruct": "#aec7e8",
    "gibbs": "#d62728",
    "gibbs_instruct": "#9467bd",
    "barker_gibbs": "#e377c2",
    "gambling_gibbs": "#98df8a",
}


def _normalize_method(method: str) -> str:
    if method.endswith("_reasoning"):
        return method[: -len("_reasoning")]
    return method


def parse_experiment(model: str, name: str) -> AlgoKey:
    """Return (model, method, temp, gamma).

    ``temp`` and ``gamma`` are ``None`` for uninformative priors.
    """
    name = RUN_SUFFIX_PATTERN.sub("", name)

    if model == "uninformative":
        if name == "uniform":
            return (model, "uniform", None, None)
        if name == "fair":
            return (model, "fair", None, None)
        if EDGE_BETA_PATTERN.match(name) or name == "edge":
            return (model, "edge", None, None)
        return (model, name, None, None)

    m = LLM_EXP_PATTERN.match(name)
    if m:
        method = _normalize_method(m.group(1))
        return (model, method, float(m.group(2)), float(m.group(3)))
    return (model, _normalize_method(name), None, None)


def load_results(base_dir: Path) -> dict[AlgoKey, list[dict]]:
    """Walk ``base_dir/<model>/<experiment>/results.json`` and group runs."""
    grouped: dict[AlgoKey, list[dict]] = defaultdict(list)
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name != "uninformative" and slug_to_hf(model_dir.name) is None:
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
        model, method, temp, gamma_key = key

        if model == "uninformative":
            if method in METHOD_DISPLAY:
                merged[key].extend(runs)
            continue

        if model not in family_slugs:
            continue
        if gamma_key is not None and gamma_key != gamma:
            continue

        plot_method = canonical_method(method, model)
        if plot_method not in METHOD_DISPLAY:
            continue

        plot_key = (family_base, plot_method, temp, gamma_key)
        merged[plot_key].extend(runs)

    return dict(merged)


def _model_sort_idx(model: str) -> tuple[int, str]:
    # Uninformative firs then LLMs alphabetically.
    return (0 if model == "uninformative" else 1, model)


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
            print(f"{k}: {metric_label}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.close(fig)


def plot_family(
    base: Path,
    dataset_name: str,
    family_id: str,
    family_slugs: tuple[str, ...],
    gamma: float,
) -> None:
    family_base = hf_to_slug(family_id)
    available_slugs = [slug for slug in family_slugs if (base / slug).is_dir()]

    grouped = load_results(base)
    grouped = group_family_results(
        grouped,
        family_slugs=frozenset(family_slugs),
        family_base=family_base,
        gamma=gamma,
    )

    if not grouped:
        raise FileNotFoundError(
            f"No matching results under {base} for family={family_id!r}, gamma={gamma}"
        )

    print(
        f"Loaded algorithms (uninformative + {family_id}, "
        f"slugs={available_slugs}, gamma={gamma}):"
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
    make_boxplot(grouped, metrics, title=title, save_path=base / f"{plot_stem}.png")
    make_boxplot(grouped, metrics, title=title, save_path=base / f"{plot_stem}.pdf")


def main(args):
    base = BASE_DIR / args.dataset_name / f"n{args.n_samples}"
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")

    families = families_with_results(base)
    if not families:
        available = sorted(p.name for p in base.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"No known model directories found under {base}. Available: {available}"
        )

    for family_id, family_slugs in families:
        try:
            plot_family(
                base,
                dataset_name=args.dataset_name,
                family_id=family_id,
                family_slugs=family_slugs,
                gamma=args.gamma,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {args.dataset_name} {family_id} gamma={args.gamma}: {exc}")


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
        args.dataset_name = dataset
        for gamma in args.gammas:
            args.gamma = gamma
            try:
                main(args)
            except FileNotFoundError as exc:
                print(f"Skipping {dataset} gamma={gamma}: {exc}")
                continue
