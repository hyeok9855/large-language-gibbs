"""LaTeX table of DAT scores (mean +/- std over scorable answers, with the
scorable count) per model and method. Run evaluate.py first."""

import argparse
import json
from pathlib import Path

from divergent_association_task.utils import RESULTS_DIR

MODELS = [
    ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B Base"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B Instruct"),
    ("meta-llama/Llama-3.1-70B", "Llama-3.1-70B Base"),
    ("allenai/Olmo-3-1125-32B", "OLMo-3-32B Base"),
    ("allenai/Olmo-3.1-32B-Instruct", "OLMo-3-32B Instruct"),
    ("google/gemma-4-31B", "gemma-4-31B Base"),
    ("google/gemma-4-31B-it", "gemma-4-31B Instruct"),
    ("Qwen/Qwen3.8-27B", "Qwen3.8-27B Instruct"),
]
FORMATS = [
    ("numbered", r"\texttt{1.\ 2.}"),
    ("bullet", r"\texttt{-}"),
    ("comma", r"\texttt{,}"),
    ("bracket", r"\texttt{[a, b,}"),
]


def cell(entries: list[dict], model: str, method: str) -> str:
    match = [e for e in entries if e["model_name"] == model and e["method"] == method]
    if not match or match[0]["mean"] is None:
        return "--"
    e = match[0]
    return f"{e['mean']:.2f} $\\pm$ {e['std']:.2f} ({e['n_valid']}/{e['n_answers']})"


def main(summary_path: Path, tag: str) -> None:
    entries = json.loads(summary_path.read_text())
    print(r"\begin{tabular}{ll" + "c" * len(FORMATS) + "}")
    print(r"\toprule")
    print("Model & Method & " + " & ".join(label for _, label in FORMATS) + r" \\")
    print(r"\midrule")
    for model, label in MODELS:
        direct = [cell(entries, model, f"direct_{f}{tag}") for f, _ in FORMATS]
        gibbs = [cell(entries, model, f"gibbs_{f}{tag}") for f, _ in FORMATS]
        print(f"{label} & Direct & " + " & ".join(direct) + r" \\")
        print(" & Gibbs & " + " & ".join(gibbs) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "summary.json")
    parser.add_argument("--prefix", action="store_true", help="table of the --answer_prefix runs")
    args = parser.parse_args()
    main(args.summary, "_prefix" if args.prefix else "")
