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
