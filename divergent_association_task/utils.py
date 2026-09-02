import re
from pathlib import Path

DAT_DIR = Path(__file__).parent
ASSETS_DIR = DAT_DIR / "assets"
RESULTS_DIR = DAT_DIR / "results"

# Chat-template kwargs per instruct model. Explicit because Qwen3.8's template
# turns thinking *on* when the flag is absent.
MODEL_CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    "Qwen/Qwen3.8-27B": {"enable_thinking": False},
    "google/gemma-4-31B-it": {"enable_thinking": False},
    "meta-llama/Llama-3.1-8B-Instruct": {},
    "meta-llama/Llama-3.1-70B-Instruct": {},
    "allenai/Olmo-3.1-32B-Instruct": {},
}


def dat_prompt(n_words: int = 10) -> str:
    """The prompt of Bellemare-Pepin et al. (2024), verbatim from their DAT_GPT code
    (scripts/api_call_dat_gpt4.py, NO_STRATEGY), including the double space."""
    return (
        f"Please enter {n_words} words that are as different from each other as possible, "
        "in all meanings and uses of the words. Rules: Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no "
        "specific people or places). No specialised vocabulary (e.g., no technical "
        "terms). Think of the words on your own (e.g., do not just look at objects in "
        f"your surroundings).  Make a list of these {n_words} words, a single word in "
        "each entry of the list."
    )


_LIST_ITEM = re.compile(r"^[ \t]*(?:\d+[.):]|[-*•])[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def parse_words(text: str, n_words: int = 10) -> list[str]:
    """First n_words list entries of a free-form answer (numbered, bulleted, or one
    per line / comma-separated). Entries are stripped of markdown and trailing
    explanations; the scorer does the rest of the cleaning."""
    items = _LIST_ITEM.findall(text)
    if len(items) < n_words:  # inline numbering, or one word per line / comma
        items = re.split(r"[\n,;:]+|\s+(?=\d+[.)]\s)", text)
        items = [re.sub(r"^\s*(?:\d+[.):]|[-*•])\s*", "", p) for p in items]
    words = []
    for item in items:
        item = re.sub(r"[*_`\"'“”]", "", item)
        item = re.split(r"\s[-–—]\s|[:(]", item)[0].strip(" .,;:!")
        if item and len(item.split()) <= 3:
            words.append(item)
    return words[:n_words]


VALID_WORDS_PATH = ASSETS_DIR / "valid_words.txt"


def load_valid_words() -> set[str]:
    """The words the scorer can embed (its dictionary intersected with GloVe).
    Cached from the scoring assets on first use; the scan takes a minute."""
    if not VALID_WORDS_PATH.exists():
        from divergent_association_task import dat

        VALID_WORDS_PATH.write_text("\n".join(sorted(dat.Model().vectors)) + "\n")
    return set(VALID_WORDS_PATH.read_text().split())


def validate(word: str, valid_words: set[str]) -> str | None:
    """`dat.Model.validate`: the form the scorer will use, or None if it cannot
    embed the word at all."""
    clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
    if len(clean) <= 1:
        return None
    if " " in clean:
        candidates = [re.sub(r" +", "-", clean), re.sub(r" +", "", clean)]
    else:
        candidates = [clean] + ([clean.replace("-", "")] if "-" in clean else [])
    return next((c for c in candidates if c in valid_words), None)


def is_scorable(words: list[str], valid_words: set[str], minimum: int = 7) -> bool:
    """Whether `dat.Model.dat` returns a score: at least `minimum` distinct words
    it can embed, so at most n_words - minimum may be non-words or repeats."""
    return len({v for word in words if (v := validate(word, valid_words))}) >= minimum
