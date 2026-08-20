"""Field-by-field word continuation for DAT.

Reuses the sampling/ continuation machinery (completions API for base models,
chat assistant-prefill for instruct models, vLLM regex structured outputs) and
adds the field type DAT needs: strings constrained by a pattern. The next
tokens of the partial JSON object are constrained to ` ?"<word>"`.
"""

import json
import re
from typing import Any

from sampling.continuation_llm import ContinuationOpenAICompatLLM


def word_regex(subschema: dict[str, Any]) -> str:
    """Shape regex for the next JSON string value: optional leading space, the
    quoted word, then an optional trailing JSON delimiter.

    Both allowances are load-bearing tokenizer escapes (cf. numeric_regex):
    BPE merges ' "' on the way in, and on the way out the closing quote almost
    always arrives fused with the next delimiter ('",', '", ', '"}'). A grammar
    admitting only a bare closing quote followed by forced EOS starves those
    paths, and the model keeps emitting letters until the length cap instead -
    the first live run degenerated into 20-char concatenations
    ('booktreasuremenugrav') in most answers for exactly this reason. The
    delimiter is stripped again at parse time.
    """
    pattern = subschema.get("pattern")
    if not pattern or not (pattern.startswith("^") and pattern.endswith("$")):
        raise ValueError(f"Anchored string pattern required: {subschema}")
    # \n must stay an escape sequence: xgrammar rejects literal newlines in
    # character classes.
    return f' ?"{pattern[1:-1]}"[,}}]?[ \\n]?'


class ContinuationWordLLM(ContinuationOpenAICompatLLM):
    """Continuation client extended with pattern-constrained string fields."""

    def generate(
        self,
        prompt: str | tuple[str, str],
        subschema: dict[str, Any],
        verbose: bool = False,
        max_trials: int = 10,
    ) -> Any:
        if subschema.get("type") != "string" or "enum" in subschema:
            return super().generate(prompt, subschema, verbose, max_trials)
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1.")

        for i in range(max_trials):
            try:
                content = self._continuation_generate(
                    prompt, word_regex(subschema), verbose=verbose
                )
                word = json.loads(content.strip().rstrip(",}").strip())
                if not re.fullmatch(subschema["pattern"], word):
                    raise ValueError(f"Word {word!r} does not match {subschema['pattern']}")
                return word
            except Exception as exc:
                print(f"Error during field value generation:\n{exc}")
                if i < max_trials - 1:
                    print(f"Retrying ({i + 1}/{max_trials}) ...")

        raise RuntimeError(f"Failed to generate a valid field value after {max_trials} trials.")
