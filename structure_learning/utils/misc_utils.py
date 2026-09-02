import json
from pathlib import Path

STRUCTURE_LEARNING_DIR = Path(__file__).parent.parent
DATASETS_DIR = STRUCTURE_LEARNING_DIR / "datasets"


def load_meta(path: Path | str) -> dict:
    """Load a dataset's `meta_data.json`."""
    with open(path) as f:
        return json.load(f)
