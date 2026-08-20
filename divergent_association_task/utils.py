from pathlib import Path

# All instruct-type models run the same constrained-decoding protocol;
# --manual_reasoning (a reasoning field inside the constrained JSON, as in
# sampling/) is optional for any instruct model. It is never a substitute for a
# model's native thinking: under structured outputs the grammar starts at the
# first generated token, so a Think model cannot emit a <think> block at all.
MODEL_NAME_TO_TYPE = {
    "meta-llama/Llama-3.1-8B": "base",
    "meta-llama/Llama-3.1-8B-Instruct": "instruct",
    "meta-llama/Llama-3.1-70B": "base",
    "meta-llama/Llama-3.1-70B-Instruct": "instruct",
    "allenai/Olmo-3-1125-32B": "base",
    "allenai/Olmo-3.1-32B-Instruct": "instruct",
    "google/gemma-4-31B": "base",
    "google/gemma-4-31B-it": "instruct",
}

DAT_DIR = Path(__file__).parent
ASSETS_DIR = DAT_DIR / "assets"
RESULTS_DIR = DAT_DIR / "results"

# Single lowercase English word (letters, optional internal hyphens), 2-20 chars.
# Enforced by vLLM structured outputs; the DAT scorer re-validates against its
# dictionary anyway, so this only rules out non-words like numbers or phrases.
WORD_PATTERN = "^[a-z][a-z-]{0,18}[a-z]$"


def word_var_names(n_words: int) -> list[str]:
    """Variable names word_1, ..., word_n."""
    return [f"word_{i}" for i in range(1, n_words + 1)]


def build_schema(n_words: int) -> dict:
    """JSON schema for one DAT answer: an object with one property per word."""
    properties = {
        name: {"type": "string", "pattern": WORD_PATTERN} for name in word_var_names(n_words)
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }
