from pathlib import Path
import json

MODEL_NAME_TO_TYPE: dict[str, str] = {
    "meta-llama/Llama-3.1-8B": "base",
    "meta-llama/Llama-3.1-8B-Instruct": "instruct",
    "meta-llama/Llama-3.1-70B": "base",
    "meta-llama/Llama-3.1-70B-Instruct": "instruct",
    "allenai/Olmo-3-1125-32B": "base",
    "allenai/Olmo-3-32B-Think": "instruct",
}

STRUCTURE_LEARNING_DIR = Path(__file__).parent.parent
DATASETS_DIR = STRUCTURE_LEARNING_DIR / "datasets"


def load_meta(path: Path | str) -> dict:
    """Load a dataset's `meta_data.json`."""
    with open(path) as f:
        return json.load(f)
