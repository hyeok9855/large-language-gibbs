"""Score saved answers with the Olson et al. (2021) DAT scorer: mean pairwise GloVe
cosine distance x 100 over the first 7 valid words; answers with fewer than 7
valid words are unscorable and dropped, as in the reference protocol.
Writes results/summary.json and results/dat_summary.png. Needs assets/, no LLM."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from divergent_association_task import dat
from divergent_association_task.utils import RESULTS_DIR

HUMAN_MEAN_DAT = 78.4  # Olson et al. (2021), study 1A


def score(model: dat.Model, answers: list[list[str]]) -> list[float | None]:
    return [None if (s := model.dat(words)) is None else float(s) for words in answers]


def summarize(model: dat.Model, path: Path) -> dict:
    data = json.loads(path.read_text())
    scores = score(model, data["samples"])
    valid = [s for s in scores if s is not None]
    entry = {
        "file": path.name,
        "method": data["method"],
        "model_name": data["model_name"],
        "temperature": data["temperature"],
        "n_answers": len(scores),
        "n_valid": len(valid),
        "mean": float(np.mean(valid)) if valid else None,
        "std": float(np.std(valid)) if valid else None,
        "top_words": Counter(w.lower() for a in data["samples"] for w in a).most_common(10),
        "scores": scores,
    }
    if "chains" in data:  # DAT score along each chain, None where unscorable
        entry["trajectories"] = [score(model, chain) for chain in data["chains"]]
    return entry


def plot(entries: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = sorted({e["model_name"] for e in entries})
    fig, axes = plt.subplots(2, len(models), figsize=(3.2 * len(models), 6.5), squeeze=False)
    for ax_box, ax_traj, name in zip(axes[0], axes[1], models):
        mine = [e for e in entries if e["model_name"] == name]
        ax_box.boxplot(
            [[s for s in e["scores"] if s is not None] for e in mine],
            tick_labels=[f"{e['method']}\n{e['n_valid']}/{e['n_answers']}" for e in mine],
        )
        ax_box.axhline(HUMAN_MEAN_DAT, ls="--", c="gray", lw=0.8)
        ax_box.set_title(name.split("/")[-1], fontsize=9)
        for e in mine:
            if "trajectories" not in e:
                continue
            grid = np.array(e["trajectories"], dtype=float)  # None -> nan
            with np.errstate(all="ignore"):
                ax_traj.plot(np.nanmean(grid, axis=0), c="C0")
            ax_traj.set_xlabel("Gibbs step")
            ax_traj.set_ylabel("mean DAT of scorable states", color="C0")
            ax_valid = ax_traj.twinx()
            ax_valid.plot(np.mean(~np.isnan(grid), axis=0), c="C1", lw=0.8)
            ax_valid.set_ylim(-0.02, 1.02)
            ax_valid.set_ylabel("fraction scorable", color="C1")
    axes[0][0].set_ylabel("DAT score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main(results_dir: Path, do_plot: bool) -> None:
    print("Loading GloVe...")
    model = dat.Model()
    paths = [p for p in sorted(results_dir.rglob("*.json")) if p.name != "summary.json"]
    entries = [summarize(model, p) for p in paths]
    print(f"\n{'model':<34} {'method':<8} {'valid':>9} {'DAT':>16}")
    for e in entries:
        dat_str = f"{e['mean']:.2f} +/- {e['std']:.2f}" if e["mean"] is not None else "-"
        valid = f"{e['n_valid']}/{e['n_answers']}"
        print(f"{e['model_name']:<34} {e['method']:<8} {valid:>9} {dat_str:>16}")
    print(f"Human mean (Olson et al. 2021): {HUMAN_MEAN_DAT}")
    (results_dir / "summary.json").write_text(json.dumps(entries, indent=1))
    if do_plot and entries:
        plot(entries, results_dir / "dat_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args.results_dir, args.plot)
