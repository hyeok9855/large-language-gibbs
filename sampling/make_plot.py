import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sampling.targets import TARGETS, get_target
from sampling.utils import indexed_var_names

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _multinomial_sum_caption(samples, params):
    if not samples or not isinstance(samples[0], dict):
        return None
    names = indexed_var_names(params["n_vars"])
    n_trials = params["n_trials"]
    totals = [sum(float(sample[name]) for name in names) for sample in samples]
    frac = float(np.mean(np.isclose(totals, n_trials)))
    return f"P(sum={n_trials})={frac:.3f}, mean sum={np.mean(totals):.2f}"


# Display order and labels for methods. Each entry is a regex matching the raw
# method name (as parsed from filenames) and a label template formatted with
# any named capture groups.
METHOD_DISPLAY_PATTERNS = [
    (re.compile(r"^independent(?:_reasoning)?$"), "Independent"),
    (re.compile(r"^batch(?:_reasoning)?(?:_nc\d+)?$"), "Batch"),
    (
        re.compile(r"^direct(?:_reasoning)?_k(?P<k>\d+)(?:_nc\d+)?$"),
        "Direct (K={k})",
    ),
    (
        re.compile(r"^direct_fixed(?:_reasoning)?_k(?P<k>\d+)(?:_nc\d+)?$"),
        "Direct-fixed (K={k})",
    ),
    (
        re.compile(r"^direct_continuation_k(?P<k>\d+)(?:_nc\d+)?$"),
        "Direct-conti. (K={k})",
    ),
    (
        re.compile(r"^direct_fixed_continuation_k(?P<k>\d+)(?:_nc\d+)?$"),
        "Direct-fixed-conti. (K={k})",
    ),
    (
        re.compile(r"^gibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Gibbs (K={k}, B={b})",
    ),
    (
        re.compile(r"^gibbs_continuation_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Gibbs-conti. (K={k}, B={b})",
    ),
    (
        re.compile(r"^barkergibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Barker-Gibbs (K={k}, B={b})",
    ),
    (
        re.compile(r"^gamblinggibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
        "Gambling-Gibbs (K={k}, B={b})",
    ),
]


def get_method_display(method):
    """Return ((primary_order, k, b), display_label) for a method.

    Methods not matching any known pattern sort to the end alphabetically and
    fall back to their raw name as the label.
    """
    for order, (pattern, label_template) in enumerate(METHOD_DISPLAY_PATTERNS):
        m = pattern.fullmatch(method)
        if m:
            groups = m.groupdict()
            label = label_template.format(**groups) if groups else label_template
            k = int(groups["k"]) if groups.get("k") else 0
            b = int(groups["b"]) if groups.get("b") else 0
            return (order, k, b), label
    return (len(METHOD_DISPLAY_PATTERNS), 0, 0), method


def matches_method_display_pattern(method):
    return any(pattern.fullmatch(method) for pattern, _ in METHOD_DISPLAY_PATTERNS)


def is_reasoning_method(method):
    return "_reasoning" in method


def sort_methods(methods):
    return sorted(methods, key=lambda m: (get_method_display(m)[0], m))


def parse_filename(filename):
    basename = Path(filename).stem

    seed_match = re.search(r"_seed(\d+)", basename)
    if not seed_match:
        return "Unknown", None

    seed = int(seed_match.group(1))
    method = re.sub(r"_seed\d+", "", basename)

    return method, seed


def parse_parameter_dir(target_name, dirname):
    if target_name not in TARGETS:
        return None
    return get_target(target_name).parse_dir_name(dirname)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-unknown-methods",
        action="store_true",
        help="Ignore runs whose method name does not match METHOD_DISPLAY_PATTERNS.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = RESULTS_DIR
    if not base_dir.is_dir():
        print(f"No results directory found at {base_dir}.")
        return

    for target_dir in base_dir.iterdir():
        if not target_dir.is_dir():
            continue

        found = False
        for data_dir in sorted(target_dir.iterdir()):
            if not data_dir.is_dir():
                continue

            params = parse_parameter_dir(target_dir.name, data_dir.name)
            if params is None:
                print(f"Skipping unrecognized result directory {data_dir}.")
                continue

            found = True
            plot_result_dir(
                target_dir.name,
                data_dir,
                params,
                ignore_unknown_methods=args.ignore_unknown_methods,
            )

        if not found:
            print(f"No parameterized result directories found in {target_dir}.")


def plot_exp_dir(target_name, exp_dir, params, method_data, plot_suffix=""):
    methods = sort_methods(method_data.keys())
    method_labels = {m: get_method_display(m)[1] for m in methods}

    if not methods:
        return

    target = get_target(target_name)
    bin_edges = target.bin_edges(params)
    bin_widths = np.diff(bin_edges)

    fig, axes = plt.subplots(
        1, len(methods), figsize=(6 * len(methods), 4), sharex=True, sharey=False
    )
    if len(methods) == 1:
        axes = [axes]

    # Plot Max Autocorrelation per Seed (lags 1..max_lag; lag 0 excluded since it's trivially 1)
    max_lag = 128
    fig_max_acf, axes_max_acf = plt.subplots(
        len(methods), 1, figsize=(8, 3 * len(methods)), sharex=False, sharey=True
    )
    if len(methods) == 1:
        axes_max_acf = [axes_max_acf]

    max_acfs_dict = {}
    for idx, method in enumerate(methods):
        ax = axes_max_acf[idx]
        label = method_labels[method]
        seeds = sorted(method_data[method].keys())
        if not seeds:
            ax.set_title(f"{label} (No Data)", fontweight="bold", fontsize=9)
            continue

        plotted_seeds = []
        max_abs_acfs = []
        for seed in seeds:
            data = target.mixing_values(method_data[method][seed], params)
            n = len(data)
            if n < 2:
                continue
            x = np.array(data, dtype=float) - np.mean(data)
            var = np.var(data)
            if var <= 0:
                continue
            acf = np.correlate(x, x, mode="full")[-n:]
            acf = acf / (var * n)
            upper = min(len(acf), max_lag + 1)
            acf_window = acf[1:upper]
            if len(acf_window) == 0:
                continue
            max_abs_acfs.append(float(np.max(np.abs(acf_window))))
            plotted_seeds.append(seed)

        if not max_abs_acfs:
            ax.set_title(f"{label} (No Data)", fontweight="bold", fontsize=9)
            continue

        positions = np.arange(len(plotted_seeds))
        ax.bar(
            positions,
            max_abs_acfs,
            color="teal",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(plotted_seeds, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Max |ACF|", fontweight="bold", fontsize=8)
        ax.set_title(
            f"{label} Max |ACF| over lags 1-{max_lag}",
            fontweight="bold",
            fontsize=9,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        max_acfs_dict[method] = max(max_abs_acfs)

    axes_max_acf[-1].set_xlabel("Trial (Seed)", fontweight="bold", fontsize=8)
    plt.tight_layout()
    out_file_max_acf = exp_dir / f"combined_max_autocorrelation{plot_suffix}.png"
    fig_max_acf.savefig(out_file_max_acf)
    plt.close(fig_max_acf)

    print(f"Saved combined max autocorrelation plot to {out_file_max_acf}")

    for idx, method in enumerate(methods):
        ax = axes[idx]
        label = method_labels[method]
        seeds = list(method_data[method].keys())
        if not seeds:
            ax.set_title(f"{label} Sampling Distribution (No Data)")
            print(f"No data found for {method} sampling.")
            continue

        histograms = []
        for seed in seeds:
            data = target.histogram_values(method_data[method][seed], params)
            counts, _ = np.histogram(data, bins=bin_edges, density=True)
            histograms.append(counts)

        histograms = np.array(histograms)
        mean_counts = np.mean(histograms, axis=0)
        std_counts = np.std(histograms, axis=0)
        # clip mean_count - std_count to be non-negative
        std_counts = np.where(mean_counts - std_counts > 0, std_counts, mean_counts)

        # If there's only 1 seed, standard deviation is 0, so errorbars won't show
        ax.bar(
            bin_edges[:-1],
            mean_counts,
            yerr=std_counts,
            width=bin_widths,
            align="edge",
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            error_kw=dict(ecolor="darkblue", lw=1, capsize=0),
            label="Empirical Mean",
        )

        ref_x, ref_y = target.reference(params)
        ax.plot(ref_x, ref_y, color="r", linestyle="--", label="True Distribution")
        ax.set_xlim(*target.xlim(params))

        if idx == 0:
            ax.set_ylabel("Empirical Density", fontsize=20)

        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

        ax.set_title(label, fontsize=20)
        max_acf = max_acfs_dict.get(method)
        ax.text(
            0.77,
            0.87,
            f"max |ACF|: {max_acf:.2f}" if max_acf is not None else "max |ACF|: n/a",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            color="black",
            backgroundcolor="lightgray",
        )

        if target.name == "multinomial":
            raw = [sample for seed in seeds for sample in method_data[method][seed]]
            diag = _multinomial_sum_caption(raw, params)
            if diag:
                print(f"{label}: {diag}")
                ax.text(
                    0.50,
                    0.02,
                    diag,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="black",
                )

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    out_file = exp_dir / f"combined_histogram{plot_suffix}.png"
    fig.savefig(out_file)
    plt.close(fig)

    print(f"Saved combined plot to {out_file}")


def plot_result_dir(target_name, data_dir, params, ignore_unknown_methods=False):
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir():
            continue

        method_data = {}
        for filepath in model_dir.glob("*.json"):
            method, seed = parse_filename(filepath)
            if method == "Unknown" or seed is None:
                continue
            if ignore_unknown_methods and not matches_method_display_pattern(method):
                print(f"Skipping run with unrecognized method {method} from {filepath}.")
                continue

            with open(filepath, "r") as f:
                data = json.load(f)

            if method not in method_data:
                method_data[method] = {}
            if seed not in method_data[method]:
                method_data[method][seed] = []
            method_data[method][seed].extend(data)

        reasoning_data = {
            method: seeds for method, seeds in method_data.items() if is_reasoning_method(method)
        }
        sampling_data = {
            method: seeds
            for method, seeds in method_data.items()
            if not is_reasoning_method(method)
        }

        if sampling_data:
            plot_exp_dir(target_name, model_dir, params, sampling_data)
        if reasoning_data:
            plot_exp_dir(target_name, model_dir, params, reasoning_data, plot_suffix="_reasoning")


if __name__ == "__main__":
    main()
