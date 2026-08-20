"""Prompt templates for the Divergent Association Task (DAT).

The task instructions follow Olson et al. (2021, PNAS 118(25)) as adapted for
LLMs by Chen & Ding (2023, arXiv:2310.11158) and Bellemare-Pepin et al. (2024,
arXiv:2405.13012): produce N single-word nouns that are as different from each
other as possible. A sample is one named list (word_1, ..., word_N). The joint
prompt (observed=None) asks for all N words at once; the conditional prompt
(observed = the other words) is the Gibbs kernel: it shows the current values
of the remaining words and asks for the missing one(s) so that the *whole*
answer stays as mutually different as possible.
"""

import json
from argparse import Namespace
from typing import Any, Callable

from divergent_association_task.utils import word_var_names


def dat_instructions(n_words: int) -> str:
    return (
        f"Write {n_words} words that are as different from each other as possible, "
        "in all meanings and uses of the words. Rules: Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no "
        "specific people or places). No specialised vocabulary (e.g., no technical terms)."
    )


def create_template(args: Namespace) -> Callable[..., str]:
    instructions = dat_instructions(args.n_words)

    if args.model_type == "base":

        def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
            if observed is None:
                return f"{instructions}\nHere is an answer, formatted as JSON:\n"

            n_missing = len([key for key in schema["required"] if key != "reasoning"])
            remaining = "remaining words" if n_missing > 1 else "remaining word"
            return (
                f"{instructions}\n"
                f"Here is a partial answer:\n{json.dumps(observed)}\n"
                f"Here {'are' if n_missing > 1 else 'is'} the {remaining} of the same "
                "answer, formatted as JSON:\n"
            )

    else:

        def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
            if observed is None:
                return (
                    f"{instructions} "
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

            n_missing = len([key for key in schema["required"] if key != "reasoning"])
            remaining = "remaining words" if n_missing > 1 else "remaining word"
            return (
                f"{instructions} "
                f"You have already written these words: {json.dumps(observed)}. "
                f"Write the {remaining} so that all {args.n_words} words are as "
                "different from each other as possible. "
                f"Respond with JSON that follows this schema: {json.dumps(schema)}"
            )

    return template


def create_continuation_template(args: Namespace) -> Callable[..., str | tuple[str, str]]:
    """Named JSON-object continuation for direct_continuation and gibbs_continuation.

    Mirrors sampling/templates/joint.py: the template receives the words already
    in context plus the name of the next field, and returns a raw prompt ending
    in the partial JSON (base) or a (user, assistant-prefill) pair (instruct).
    """
    instructions = dat_instructions(args.n_words)
    var_names = word_var_names(args.n_words)
    if len(var_names) > 2:
        var_str = f"{var_names[0]}, ..., {var_names[-1]}"
    else:
        var_str = ", ".join(var_names)

    def prefill(observed: dict[str, Any] | None, next_key: str) -> str:
        items: list[str] = []
        if observed:
            items.extend(
                f"{json.dumps(key)}: {json.dumps(value)}" for key, value in observed.items()
            )
        items.append(f"{json.dumps(next_key)}:")
        return "{" + ", ".join(items)

    if args.model_type == "base":

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            if not next_key:
                raise ValueError("Continuation requires next_key.")
            return f"{instructions}\nHere is an answer, formatted as JSON:\n" + prefill(
                observed, next_key
            )

    else:

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            if not next_key:
                raise ValueError("Continuation requires next_key.")
            return (
                f"{instructions} Write the answer as a JSON object with keys ({var_str}).",
                prefill(observed, next_key),
            )

    return template
