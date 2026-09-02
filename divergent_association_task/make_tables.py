"""LaTeX table of DAT scores (mean +/- std over scorable answers, with the
scorable count) per model and method. Run evaluate.py first."""

import argparse
import json
from pathlib import Path

from divergent_association_task.utils import RESULTS_DIR

MODELS = [
    ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B Base"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B Instruct"),
    ("allenai/Olmo-3-1125-32B", "OLMo-3-32B Base"),
    ("allenai/Olmo-3.1-32B-Instruct", "OLMo-3-32B Instruct"),
    ("google/gemma-4-31B", "gemma-4-31B Base"),
    ("google/gemma-4-31B-it", "gemma-4-31B Instruct"),
    ("Qwen/Qwen3.8-27B", "Qwen3.8-27B Instruct"),
]
METHODS = [("direct", "Direct"), ("gibbs", "Gibbs")]


def cell(entries: list[dict], model: str, method: str) -> str:
    match = [e for e in entries if e["model_name"] == model and e["method"] == method]
    if not match or match[0]["mean"] is None:
        return "--"
    e = match[0]
    return f"{e['mean']:.2f} $\\pm$ {e['std']:.2f} ({e['n_valid']}/{e['n_answers']})"


def main(summary_path: Path) -> None:
    entries = json.loads(summary_path.read_text())
    print(r"\begin{tabular}{l" + "c" * len(METHODS) + "}")
    print(r"\toprule")
    print("Model & " + " & ".join(label for _, label in METHODS) + r" \\")
    print(r"\midrule")
    for model, label in MODELS:
        print(f"{label} & " + " & ".join(cell(entries, model, m) for m, _ in METHODS) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "summary.json")
    main(parser.parse_args().summary)
