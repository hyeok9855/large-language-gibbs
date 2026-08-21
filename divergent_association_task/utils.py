import re
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
VALID_WORDS_PATH = ASSETS_DIR / "valid_words.txt"


def load_valid_words() -> set[str]:
    """Lexicon for --reject_invalid: exactly the words the scorer can embed
    (dictionary intersected with GloVe, i.e. dat.Model().vectors). Built once
    from the scoring assets and cached as a small text file."""
    if not VALID_WORDS_PATH.exists():
        from divergent_association_task import dat  # heavy import: scans GloVe once

        VALID_WORDS_PATH.write_text("\n".join(sorted(dat.Model().vectors)) + "\n")
    return set(VALID_WORDS_PATH.read_text().split())


def is_valid_word(word: str, valid_words: set[str]) -> bool:
    """dat.Model.validate semantics for words that already match WORD_PATTERN
    (lowercase, no stray characters): direct hit, or hyphens dropped."""
    return word in valid_words or ("-" in word and word.replace("-", "") in valid_words)


def dup_key(word: str) -> str:
    """Canonical form for duplicate detection, mirroring the scorer's candidate
    logic for pattern-clean words: hyphens do not make a word distinct."""
    return word.replace("-", "")


_PROMPT_WORD_RE = re.compile(r'"word_\d+": "([a-z-]+)"')


def find_prompt_words(prompt: str | tuple[str, str]) -> list[str]:
    """word_* values already shown in a prompt (the observed JSON in chat
    conditionals, or the partial-JSON prefill in continuation prompts); both
    are built with json.dumps, so the quoting is exact."""
    text = prompt[1] if isinstance(prompt, tuple) else prompt
    return _PROMPT_WORD_RE.findall(text)


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
