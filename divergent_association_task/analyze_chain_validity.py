"""Why do Gibbs chains lose validity? Transition-level forensics on saved chains.

Every Gibbs result stores its full chains and the key resampled at each step,
so each conditional draw can be classified: did it produce a dictionary word or
a non-word? a duplicate? and how does that depend on how corrupted the
conditioning context already was? Also fits a birth-death sanity check: a
simulation driven only by the measured P(non-word | k bad words in context)
table should reproduce the observed drift if context feedback is the whole
story. Needs the scoring assets (download_assets.sh) but no LLM.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from divergent_association_task import dat
from divergent_association_task.utils import RESULTS_DIR, word_var_names


def analyze_dir(model: dat.Model, mdir: Path) -> None:
    cache: dict[str, str | None] = {}

    def clean(word: str) -> str | None:
        if word not in cache:
            cache[word] = model.validate(word)
        return cache[word]

    def state_stats(words: list[str]) -> tuple[int, int, int]:
        """(n_nonword, n_dup, n_unique_valid) for one state, dat()-style."""
        uniques: list[str] = []
        n_nonword = n_dup = 0
        for w in words:
            v = clean(w)
            if v is None:
                n_nonword += 1
            elif v in uniques:
                n_dup += 1
            else:
                uniques.append(v)
        return n_nonword, n_dup, len(uniques)

    print(f"\n{'=' * 78}\n{mdir.name}\n{'=' * 78}")

    # One-pass baselines: per-position non-word rate and what kills answers.
    for pattern in ["direct_k*.json", "direct_reasoning_k*.json", "direct_continuation_k*.json"]:
        files = sorted(mdir.glob(pattern))
        if not files:
            continue
        method = json.loads(files[0].read_text())["method"]
        pos_bad: Counter = Counter()
        pos_tot: Counter = Counter()
        fail_nonword = fail_dup = n_invalid = n_tot = 0
        for f in files:
            d = json.loads(f.read_text())
            for ans in d["samples"]:
                n_tot += 1
                nw, dup, uniq = state_stats(ans)
                if uniq < 7:
                    n_invalid += 1
                    if len(ans) - nw >= 7:
                        fail_dup += 1
                    else:
                        fail_nonword += 1
                for i, w in enumerate(ans):
                    pos_tot[i] += 1
                    if clean(w) is None:
                        pos_bad[i] += 1
        rate = [pos_bad[i] / pos_tot[i] for i in sorted(pos_tot)]
        print(
            f"\n[{method}] invalid answers {n_invalid}/{n_tot} "
            f"(killed by non-words: {fail_nonword}, by duplicates: {fail_dup})"
        )
        print("  per-position non-word rate:", " ".join(f"{r:.2f}" for r in rate))

    # Gibbs chains: transition-level analysis.
    for pattern in ["gibbs_k*.json", "gibbs_reasoning_k*.json", "gibbs_continuation_k*.json"]:
        files = sorted(mdir.glob(pattern))
        if not files:
            continue
        method = json.loads(files[0].read_text())["method"]
        by_ctx: dict[int, Counter] = defaultdict(Counter)
        repair: Counter = Counter()
        corrupt: Counter = Counter()
        state_traj: dict[int, list[int]] = defaultdict(list)
        for f in files:
            d = json.loads(f.read_text())
            keymap = word_var_names(d["n_words"])
            for chain, rk in zip(d["chains"], d["resampled_keys"]):
                for t, keys in enumerate(rk):
                    ki = keymap.index(keys[0])
                    ctx_clean = [clean(w) for j, w in enumerate(chain[t]) if j != ki]
                    k_bad = sum(v is None for v in ctx_clean)
                    v_new = clean(chain[t + 1][ki])
                    outcome = (
                        "nonword"
                        if v_new is None
                        else "dup"
                        if v_new in [v for v in ctx_clean if v is not None]
                        else "fresh"
                    )
                    by_ctx[k_bad][outcome] += 1
                    by_ctx[k_bad]["total"] += 1
                    if clean(chain[t][ki]) is None:
                        repair["fixed" if v_new is not None else "still_bad"] += 1
                    else:
                        corrupt["broke" if v_new is None else "stayed_ok"] += 1
                for t in range(0, len(chain), 25):
                    state_traj[t].append(state_stats(chain[t])[0])

        print(f"\n[{method}]")
        print("  P(new word = non-word | k non-words in context):")
        for k in sorted(by_ctx):
            c = by_ctx[k]
            if c["total"] >= 30:
                print(
                    f"    k={k}: nonword {c['nonword'] / c['total']:.3f}  "
                    f"dup {c['dup'] / c['total']:.3f}   (n={c['total']})"
                )
        rp = repair["fixed"] + repair["still_bad"]
        cp = corrupt["broke"] + corrupt["stayed_ok"]
        if rp:
            print(f"  repair  P(valid | old invalid): {repair['fixed'] / rp:.3f} (n={rp})")
        if cp:
            print(f"  corrupt P(invalid | old valid): {corrupt['broke'] / cp:.3f} (n={cp})")
        print("  mean non-words per state over steps:")
        for t in [0, 50, 150, 300, 550]:
            if t in state_traj:
                print(f"    step {t:>3}: {np.mean(state_traj[t]):.2f}")

        # Birth-death sanity check driven only by the measured P(nonword | k).
        p_table = {k: c["nonword"] / c["total"] for k, c in by_ctx.items() if c["total"] >= 30}
        if len(p_table) >= 3 and state_traj.get(0):
            for k in range(10):
                if k not in p_table:
                    below = [p for kk, p in p_table.items() if kk < k]
                    p_table[k] = max(below) if below else min(p_table.values())
            rng = np.random.default_rng(0)
            n_sim, n_words = 2000, 10
            states = np.zeros((n_sim, n_words), bool)
            k0 = int(round(float(np.mean(state_traj[0]))))
            for i in range(n_sim):
                states[i, rng.choice(n_words, size=k0, replace=False)] = True
            sim = {}
            order = np.array([rng.permutation(n_words) for _ in range(n_sim)])
            for t in range(551):
                pos = order[:, t % n_words]
                if t % n_words == n_words - 1:
                    order = np.array([rng.permutation(n_words) for _ in range(n_sim)])
                for i in range(n_sim):
                    k_ctx = int(states[i].sum()) - int(states[i, pos[i]])
                    states[i, pos[i]] = rng.random() < p_table[k_ctx]
                if t in (50, 150, 300, 550):
                    sim[t] = states.sum(axis=1).mean()
            print(
                "  birth-death simulation from measured P(nonword | k) alone:",
                {t: round(float(v), 2) for t, v in sim.items()},
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    print("Loading GloVe vectors (takes a minute or two)...")
    model = dat.Model()
    for mdir in sorted(args.results_dir.glob("*_temp*")):
        if mdir.is_dir():
            analyze_dir(model, mdir)
