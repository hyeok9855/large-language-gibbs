"""Score saved DAT answers with the Olson et al. (2021) reference scorer.

Reads every results JSON produced by run.py, computes the DAT score of each
answer (mean pairwise GloVe cosine distance x 100 over the first 7 valid
words), and writes summary.json next to the results plus an optional plot.
Needs the assets from download_assets.sh; does not need an LLM.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from divergent_association_task import dat
from divergent_association_task.utils import RESULTS_DIR

# Mean DAT of Olson et al. (2021)'s main human sample (study 1A, N=8,572);
# both related-works papers use this cohort as the human reference.
HUMAN_MEAN_DAT = 78.4


def score_answers(model: dat.Model, answers: list[list[str]]) -> list[float | None]:
    # model.dat returns numpy float32 (not JSON serializable), hence the cast.
    scores = [model.dat(words) for words in answers]
    return [None if s is None else float(s) for s in scores]


def summarize_file(model: dat.Model, path: Path, results_dir: Path) -> dict | None:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "samples" not in data:
        return None

    scores = score_answers(model, data["samples"])
    valid = [s for s in scores if s is not None]
    word_counts = Counter(w for answer in data["samples"] for w in answer)
    entry = {
        "file": str(path.relative_to(results_dir)),
        "method": data["method"],
        "model_name": data["model_name"],
        "temperature": data["temperature"],
        "seed": data["seed"],
        "n_answers": len(scores),
        "n_valid": len(valid),
        "mean": float(np.mean(valid)) if valid else None,
        "std": float(np.std(valid)) if valid else None,
        "min": float(np.min(valid)) if valid else None,
        "max": float(np.max(valid)) if valid else None,
        "llm_calls": data.get("llm_calls"),
        "duration_seconds": data.get("duration_seconds"),
        "top_words": word_counts.most_common(10),
        "word_counts": dict(word_counts),
        "scores": scores,
    }

    if "chains" in data:
        entry["trajectories"] = [score_answers(model, chain) for chain in data["chains"]]
        entry["burn_in"] = data.get("burn_in")
        entry["thinning"] = data.get("thinning")

    return entry


def pool_groups(entries: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for entry in entries:
        key = (entry["model_name"], entry["temperature"], entry["method"])
        groups.setdefault(key, []).append(entry)

    pooled = []
    for (model_name, temperature, method), group in sorted(groups.items()):
        scores = [s for entry in group for s in entry["scores"] if s is not None]
        n_all = sum(entry["n_answers"] for entry in group)
        words = Counter()
        for entry in group:
            words.update(entry["word_counts"])
        pooled.append(
            {
                "model_name": model_name,
                "temperature": temperature,
                "method": method,
                "seeds": sorted(entry["seed"] for entry in group),
                "n_answers": n_all,
                "n_valid": len(scores),
                "mean": float(np.mean(scores)) if scores else None,
                "std": float(np.std(scores)) if scores else None,
                "seed_means": [entry["mean"] for entry in group],
                "top_words": words.most_common(10),
            }
        )
    return pooled


def make_plot(entries: list[dict], pooled: list[dict], results_dir: Path) -> None:
    """One figure per (model, temperature) config: score boxplots (valid answers,
    with valid counts in the labels), DAT score along the chain, and the
    fraction of chain states that are valid answers (drift diagnostic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = sorted({(entry["model_name"], entry["temperature"]) for entry in entries})
    cmap = plt.get_cmap("tab10")

    for model_name, temperature in configs:
        cfg_pooled = [
            group
            for group in pooled
            if (group["model_name"], group["temperature"]) == (model_name, temperature)
        ]
        fig, (ax_box, ax_traj, ax_valid) = plt.subplots(1, 3, figsize=(15, 4.2))

        # Panel 1: score distribution per method (valid answers only)
        labels, data = [], []
        for group in cfg_pooled:
            scores = [
                s
                for entry in entries
                if (entry["model_name"], entry["temperature"], entry["method"])
                == (model_name, temperature, group["method"])
                for s in entry["scores"]
                if s is not None
            ]
            labels.append(
                f"{group['method'].replace('_', chr(10))}\n"
                f"({group['n_valid']}/{group['n_answers']} valid)"
            )
            data.append(scores)
        if data:
            ax_box.boxplot(data, tick_labels=labels, showmeans=True)
        ax_box.axhline(HUMAN_MEAN_DAT, ls="--", c="gray", lw=1, label="human avg (Olson et al.)")
        ax_box.set_ylabel("DAT score")
        ax_box.set_title(f"DAT score ({model_name.split('/')[-1]}, T={temperature})")
        ax_box.legend(fontsize=8)

        # Panels 2+3: score along the chain (valid states) and valid-state fraction
        for idx, group in enumerate(cfg_pooled):
            color = cmap(idx % 10)
            cfg_entries = [
                entry
                for entry in entries
                if (entry["model_name"], entry["temperature"], entry["method"])
                == (model_name, temperature, group["method"])
            ]
            trajectories = [t for entry in cfg_entries for t in entry.get("trajectories", [])]
            if trajectories:
                max_len = max(len(t) for t in trajectories)
                grid = np.full((len(trajectories), max_len), np.nan)
                for i, trajectory in enumerate(trajectories):
                    grid[i, : len(trajectory)] = [np.nan if s is None else s for s in trajectory]
                n_valid_states = np.sum(~np.isnan(grid), axis=0)
                with np.errstate(invalid="ignore"):
                    mean = np.where(n_valid_states > 0, np.nanmean(grid, axis=0), np.nan)
                    std = np.where(n_valid_states > 1, np.nanstd(grid, axis=0), 0.0)
                steps = np.arange(max_len)
                ax_traj.plot(steps, mean, color=color, label=group["method"])
                ax_traj.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
                ax_valid.plot(
                    steps, n_valid_states / len(trajectories), color=color, label=group["method"]
                )
            else:
                scores = [s for entry in cfg_entries for s in entry["scores"] if s is not None]
                if scores:
                    ax_traj.axhline(
                        float(np.mean(scores)), ls=":", c=color, label=f"{group['method']} (mean)"
                    )
                n_all = sum(entry["n_answers"] for entry in cfg_entries)
                if n_all:
                    ax_valid.axhline(
                        group["n_valid"] / n_all,
                        ls=":",
                        c=color,
                        label=f"{group['method']} (answers)",
                    )
        ax_traj.axhline(HUMAN_MEAN_DAT, ls="--", c="gray", lw=1, label="human avg")
        ax_traj.set_xlabel("Gibbs step (single-word updates)")
        ax_traj.set_ylabel("DAT score (valid states)")
        ax_traj.set_title("Score along the Gibbs chain")
        ax_traj.legend(fontsize=8)

        ax_valid.set_xlabel("Gibbs step (single-word updates)")
        ax_valid.set_ylabel("fraction of valid states")
        ax_valid.set_ylim(-0.03, 1.03)
        ax_valid.set_title("Chain validity (>= 7 scorable words)")
        ax_valid.legend(fontsize=8)

        if len(configs) == 1:
            out_path = results_dir / "dat_summary.png"
        else:
            tag = f"{model_name.split('/')[-1]}_temp{temperature}"
            out_path = results_dir / f"dat_summary_{tag}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {out_path}")


def main(args: argparse.Namespace):
    print("Loading GloVe vectors (takes a minute or two)...")
    t_start = time.time()
    model = dat.Model()
    print(f"Loaded {len(model.vectors)} word vectors in {time.time() - t_start:.0f}s.")

    entries = []
    for path in sorted(args.results_dir.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            entry = summarize_file(model, path, args.results_dir)
        except (json.JSONDecodeError, KeyError, OSError, UnicodeDecodeError) as e:
            print(f"Skipping {path}: {e}")
            continue
        if entry is not None:
            entries.append(entry)
    if not entries:
        print(f"No result files found under {args.results_dir}.")
        return

    pooled = pool_groups(entries)

    header = f"{'method':<20} {'model':<28} {'temp':>5} {'seeds':>6} {'valid':>10} {'DAT':>18}"
    print("\n" + header + "\n" + "-" * len(header))
    for group in pooled:
        valid_cell = f"{group['n_valid']}/{group['n_answers']}"
        dat_cell = (
            f"{group['mean']:.2f} +/- {group['std']:.2f}"
            if group["mean"] is not None
            else "(no valid answers)"
        )
        print(
            f"{group['method']:<20} {group['model_name'].split('/')[-1]:<28} "
            f"{group['temperature']:>5} {len(group['seeds']):>6} {valid_cell:>10} {dat_cell:>18}"
        )
    print(f"\nHuman average (Olson et al. 2021): {HUMAN_MEAN_DAT}")

    summary_path = args.results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"groups": pooled, "files": entries}, f, indent=2)
    print(f"Saved summary to {summary_path}")

    if args.plot:
        make_plot(entries, pooled, args.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score DAT answers with GloVe.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plot", action="store_true")
    main(parser.parse_args())
