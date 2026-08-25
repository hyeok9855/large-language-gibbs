import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sampling.targets import TARGETS, get_target
from sampling.utils import RESULTS_DIR

MAX_LAG = 128


def get_method_display(method, patterns):
    """Return ((primary_order, k, b), display_label) for a method.

    Methods not matching any pattern in ``patterns`` sort to the end
    alphabetically and fall back to their raw name as the label.
    """
    for order, (pattern, label_template) in enumerate(patterns):
        m = pattern.fullmatch(method)
        if m:
            groups = m.groupdict()
            label = label_template.format(**groups) if groups else label_template
            k = int(groups["k"]) if groups.get("k") else 0
            b = int(groups["b"]) if groups.get("b") else 0
            return (order, k, b), label
    return (len(patterns), 0, 0), method


# --- Result parsing ---------------------------------------------------------
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


# --- Metrics ----------------------------------------------------------------
def reference_bin_mass(target, params, bin_edges):
    """Target mass per bin: the red overlay integrated bin-by-bin."""
    ref_x, ref_y = target.reference(params)
    widths = np.diff(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if len(ref_x) == len(centers) and np.allclose(ref_x, centers):
        return ref_y * widths

    mass = np.empty(len(centers))
    for i in range(len(centers)):
        grid = np.linspace(bin_edges[i], bin_edges[i + 1], 65)
        mass[i] = np.trapezoid(np.interp(grid, ref_x, ref_y, left=0.0, right=0.0), grid)
    return mass


def total_variation(empirical_mass, reference_mass, empirical_tail=0.0, reference_tail=0.0):
    """TV over the plotted bins plus one overflow cell for everything outside."""
    inside = np.abs(np.asarray(empirical_mass) - np.asarray(reference_mass)).sum()
    return 0.5 * float(inside + abs(float(empirical_tail) - float(reference_tail)))


def max_abs_acf(values, max_lag=MAX_LAG):
    """Largest |autocorrelation| over lags 1..max_lag, or None if undefined."""
    n = len(values)
    if n < 2:
        return None
    x = np.asarray(values, dtype=float) - np.mean(values)
    var = np.var(values)
    if var <= 0:
        return None
    acf = np.correlate(x, x, mode="full")[-n:] / (var * n)
    window = acf[1 : min(len(acf), max_lag + 1)]
    if len(window) == 0:
        return None
    return float(np.max(np.abs(window)))


# --- Metrics output ---------------------------------------------------------
METRICS_HEADER = (
    f"{'method':28s} {'TV':>7s} {'+-':>6s} "
    f"{'max|ACF|':>8s} {'mean|ACF|':>9s} {'seeds':>5s} {'draws':>7s} {'out%':>5s}"
)

METRICS_LEGEND = """\
TV        per-seed total variation from the target, mean +- std over seeds.
          0 = exact match, 1 = disjoint support.
max|ACF|  max |autocorrelation| over lags 1-{max_lag}, maxed over seeds. High = stuck.
mean|ACF| the same, averaged over seeds instead of maxed.
draws     scalar values pooled into the histogram (samples x coordinates).
out%      draws outside the plot window; legal, and charged to TV.
"""


def write_metrics_table(out_path, target_name, params, metrics_rows):
    """Per-method TV / autocorrelation table for one experiment."""
    lines = [
        f"# {target_name}  {params}",
        "",
        METRICS_HEADER,
        "-" * len(METRICS_HEADER),
    ]
    for row in metrics_rows:
        lines.append(
            f"{row['label']:28s} {row['tv']:7.3f} {row['tv_std']:6.3f} "
            f"{row['max_acf']:8.2f} {row['mean_acf']:9.2f} "
            f"{row['n_seeds']:5d} {row['n_values']:7d} {100 * row['oob_frac']:5.1f}"
        )

    ranked = sorted(metrics_rows, key=lambda r: r["tv"])
    lines += ["", "Ranked by TV (closest to the target first):"]
    lines += [f"  {i + 1:2d}. {r['label']:28s} {r['tv']:.3f}" for i, r in enumerate(ranked)]
    lines += ["", METRICS_LEGEND.format(max_lag=MAX_LAG)]

    out_path.write_text("\n".join(lines) + "\n")


def write_metrics_summary(out_path, all_rows):
    """One line per (experiment, method), so all cells are grep-able at once."""

    # Size label columns to their longest entry, else the numbers misalign.
    def width(header_text, values):
        return max(len(header_text), *(len(v) for v in values))

    w_target = width("target", [k[0] for k in all_rows] or [""])
    w_params = width("params", [k[1] for k in all_rows] or [""])
    w_model = width("model", [k[2] for k in all_rows] or [""])
    w_method = width("method", [r["label"] for rows in all_rows.values() for r in rows] or [""])

    header = (
        f"{'target':{w_target}s} {'params':{w_params}s} {'model':{w_model}s} "
        f"{'method':{w_method}s} {'TV':>7s} {'+-':>6s} {'rank':>4s} {'max|ACF|':>8s} "
        f"{'mean|ACF|':>9s} {'seeds':>5s} {'out%':>5s}"
    )
    lines = [header, "-" * len(header)]
    for (target_name, param_dir, model), rows in sorted(all_rows.items()):
        order = {r["method"]: i for i, r in enumerate(sorted(rows, key=lambda r: r["tv"]))}
        for row in rows:
            lines.append(
                f"{target_name:{w_target}s} {param_dir:{w_params}s} {model:{w_model}s} "
                f"{row['label']:{w_method}s} {row['tv']:7.3f} {row['tv_std']:6.3f} "
                f"{order[row['method']] + 1:4d} {row['max_acf']:8.2f} {row['mean_acf']:9.2f} "
                f"{row['n_seeds']:5d} {100 * row['oob_frac']:5.1f}"
            )
    lines += ["", METRICS_LEGEND.format(max_lag=MAX_LAG)]
    lines += [
        "rank      position within this experiment, by TV (1 = best).",
    ]
    out_path.write_text("\n".join(lines) + "\n")


# --- Plotting ---------------------------------------------------------------
def plot_exp_dir(target_name, exp_dir, params, method_data, patterns, plot_suffix=""):
    methods = sorted(method_data.keys(), key=lambda m: (get_method_display(m, patterns)[0], m))
    method_labels = {m: get_method_display(m, patterns)[1] for m in methods}

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

    # Plot Max Autocorrelation per Seed (lags 1..MAX_LAG; lag 0 excluded since it's trivially 1)
    fig_max_acf, axes_max_acf = plt.subplots(
        len(methods), 1, figsize=(8, 3 * len(methods)), sharex=False, sharey=True
    )
    if len(methods) == 1:
        axes_max_acf = [axes_max_acf]

    max_acfs_dict = {}
    acf_stats = {}
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
            acf_max = max_abs_acf([float(value) for value in method_data[method][seed]], MAX_LAG)
            if acf_max is None:
                continue
            max_abs_acfs.append(acf_max)
            plotted_seeds.append(seed)

        acf_stats[method] = max_abs_acfs

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
            f"{label} Max |ACF| over lags 1-{MAX_LAG}",
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

    # Target mass per bar, plus its own off-plot tail (the overflow reference).
    ref_mass = reference_bin_mass(target, params, bin_edges)
    ref_tail = max(0.0, 1.0 - float(ref_mass.sum()))
    metrics_rows = []

    for idx, method in enumerate(methods):
        ax = axes[idx]
        label = method_labels[method]
        seeds = list(method_data[method].keys())
        if not seeds:
            ax.set_title(f"{label} Sampling Distribution (No Data)")
            print(f"No data found for {method} sampling.")
            continue

        histograms = []
        per_seed_tv = []
        n_values = 0
        n_in_range = 0
        for seed in seeds:
            data = [float(value) for value in method_data[method][seed]]
            values = np.asarray(data, dtype=float)
            counts, _ = np.histogram(values, bins=bin_edges)
            mass = counts / len(values) if len(values) else counts.astype(float)
            tail = max(0.0, 1.0 - float(mass.sum()))
            histograms.append(mass / bin_widths)
            per_seed_tv.append(total_variation(mass, ref_mass, tail, ref_tail))
            n_values += len(values)
            n_in_range += int(counts.sum())

        histograms = np.array(histograms)
        mean_counts = np.mean(histograms, axis=0)
        std_counts = np.std(histograms, axis=0)
        # clip mean_count - std_count to be non-negative
        std_counts = np.where(mean_counts - std_counts > 0, std_counts, mean_counts)

        metrics_rows.append(
            {
                "method": method,
                "label": label,
                "tv": float(np.mean(per_seed_tv)),
                "tv_std": float(np.std(per_seed_tv)),
                "max_acf": max(acf_stats.get(method) or [float("nan")]),
                "mean_acf": (
                    float(np.mean(acf_stats[method])) if acf_stats.get(method) else float("nan")
                ),
                "n_seeds": len(seeds),
                "n_values": n_values,
                "oob_frac": 1.0 - (n_in_range / n_values) if n_values else float("nan"),
            }
        )

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

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    out_file = exp_dir / f"combined_histogram{plot_suffix}.png"
    fig.savefig(out_file)
    plt.close(fig)

    print(f"Saved combined plot to {out_file}")

    out_file_metrics = exp_dir / f"metrics{plot_suffix}.txt"
    write_metrics_table(out_file_metrics, target_name, params, metrics_rows)
    print(f"Saved metrics table to {out_file_metrics}")

    return metrics_rows


def sweep_suffix(method):
    """Which sweep a method name belongs to."""
    if "_reasoning" in method:
        return "_reasoning"
    return ""


def plot_result_dir(target_name, data_dir, params, patterns, plot_suffixes, ignore_unknown):
    """Plot every model in one parameter dir of one results tree.

    Returns metrics rows keyed by (plot_suffix, param dir, model dir).
    """
    collected = {}
    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        method_data = {}
        for filepath in sorted(model_dir.glob("*.json")):
            method, seed = parse_filename(filepath)
            if method == "Unknown" or seed is None:
                continue
            if ignore_unknown and not any(p.fullmatch(method) for p, _ in patterns):
                print(f"Skipping run with unrecognized method {method} from {filepath}.")
                continue

            with open(filepath, "r") as f:
                data = json.load(f)

            method_data.setdefault(method, {}).setdefault(seed, []).extend(data)

        for suffix in plot_suffixes:
            sweep_data = {
                method: seeds
                for method, seeds in method_data.items()
                if sweep_suffix(method) == suffix
            }
            if not sweep_data:
                continue
            rows = plot_exp_dir(
                target_name, model_dir, params, sweep_data, patterns, plot_suffix=suffix
            )
            if rows:
                collected[(suffix, data_dir.name, model_dir.name)] = rows

    return collected


def run(
    results_dir: Path,
    patterns: list[tuple[re.Pattern, str]],
    plot_suffixes: list[str],
    ignore_unknown: bool = False,
):
    if not results_dir.is_dir():
        print(f"No results directory found at {results_dir}.")
        return

    # {plot_suffix: {(target, param dir, model dir): metrics rows}}
    summaries = {suffix: {} for suffix in plot_suffixes}

    for target_dir in sorted(results_dir.iterdir()):
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
            collected = plot_result_dir(
                target_dir.name, data_dir, params, patterns, plot_suffixes, ignore_unknown
            )
            for (suffix, param_dir, model), rows in collected.items():
                summaries[suffix][(target_dir.name, param_dir, model)] = rows

        if not found:
            print(f"No parameterized result directories found in {target_dir}.")

    for suffix, all_rows in summaries.items():
        if not all_rows:
            continue
        out_path = results_dir / f"metrics_summary{suffix}.txt"
        write_metrics_summary(out_path, all_rows)
        print(f"Saved metrics summary to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Ignore runs whose method name does not match METHOD_PATTERNS.",
    )
    args = parser.parse_args()

    PATTERNS = [
        (re.compile(r"^independent$"), "Independent"),
        (re.compile(r"^batch(?:_nc\d+)?$"), "Batch"),
        (
            re.compile(r"^direct_k(?P<k>\d+)(?:_nc\d+)?$"),
            "Direct",
        ),
        (
            re.compile(r"^gibbs_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
            "Gibbs (B={b})",
        ),
        (
            re.compile(r"^barkergibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
            "Barker-Gibbs (B={b})",
        ),
        (
            re.compile(r"^gamblinggibbs(?:_reasoning)?_k(?P<k>\d+)_b(?P<b>\d+)(?:_nc\d+)?$"),
            "Gambling-Gibbs (B={b})",
        ),
    ]

    PLOT_SUFFIXES = ("", "_reasoning")

    run(
        results_dir=RESULTS_DIR,
        patterns=PATTERNS,
        plot_suffixes=PLOT_SUFFIXES,
        ignore_unknown=args.ignore_unknown,
    )
