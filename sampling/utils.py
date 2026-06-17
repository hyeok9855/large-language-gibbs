from pathlib import Path

MODEL_NAME_TO_TYPE = {
    "meta-llama/Llama-3.1-8B": "base",
    "meta-llama/Llama-3.1-8B-Instruct": "instruct",
    "meta-llama/Llama-3.1-70B": "base",
    "meta-llama/Llama-3.1-70B-Instruct": "instruct",
    "allenai/Olmo-3-1125-32B": "base",
    "allenai/Olmo-3-32B-Think": "instruct",
    "google/gemma-4-31B": "base",
    "google/gemma-4-31B-it": "instruct",
}

RESULTS_DIR = Path(__file__).parent / "results"


def round_dict(d, precision=2):
    return {k: round(v, precision) for k, v in d.items()}
