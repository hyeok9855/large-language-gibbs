"""LLM client for field-by-field sampling."""

from __future__ import annotations

import json
from typing import Any

from priorbot.llm import OpenAICompatLLM, _check_json_schema_value

MAX_FRACTION_DIGITS = 6


def numeric_regex(subschema: dict[str, Any]) -> str:
    """Shape regex for the next numeric value: optional space, optional minus,
    digit budget sized from the bounds."""
    lo = subschema.get("minimum")
    hi = subschema.get("maximum")
    if lo is None or hi is None:
        raise ValueError(f"minimum and maximum required: {subschema}")
    n_digits = max(len(str(abs(int(lo)))), len(str(abs(int(hi)))))
    sign = "-?" if lo < 0 else ""
    body = rf"{sign}\d{{1,{n_digits}}}"
    if subschema.get("type") == "number":
        body += rf"(\.\d{{1,{MAX_FRACTION_DIGITS}}})?"
    return rf" ?{body}"


class ContinuationOpenAICompatLLM(OpenAICompatLLM):
    """LLM client for field-by-field JSON continuation on base or instruct models."""

    def __init__(self, model_name: str, base_url: str, system_prompt: str = "", **kwargs):
        super().__init__(model_name, base_url, system_prompt, **kwargs)

    def generate(
        self,
        prompt: str | tuple[str, str],
        subschema: dict[str, Any],
        verbose: bool = False,
        max_trials: int = 10,
    ) -> Any:
        """Generate one field value from a partial JSON array."""
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1.")

        val_type = subschema.get("type")
        is_choice = val_type == "string" and "enum" in subschema
        if not is_choice and val_type not in ("integer", "number"):
            raise ValueError(f"Unsupported subschema for continuation generation: {subschema}")

        for i in range(max_trials):
            try:
                if is_choice:
                    choices = list(subschema["enum"])
                    return self._generate_choice(prompt, choices, verbose=verbose)
                return self._generate_numeric(prompt, subschema, verbose=verbose)
            except Exception as exc:
                print(f"Error during field value generation:\n{exc}")
                if i < max_trials - 1:
                    print(f"Retrying ({i + 1}/{max_trials}) ...")

        raise RuntimeError(f"Failed to generate a valid field value after {max_trials} trials.")

    def _generate_choice(
        self,
        prompt: str | tuple[str, str],
        choices: list[str],
        verbose: bool = False,
    ) -> str:
        content = self._continuation_generate(prompt, choices, verbose=verbose)
        if content not in choices:
            raise ValueError(f"Response {content!r} is not in choice set {choices}")
        return content

    def _generate_numeric(
        self,
        prompt: str | tuple[str, str],
        subschema: dict[str, Any],
        verbose: bool = False,
    ) -> Any:
        """Generate a numeric value continuing the partial JSON array/object.

        Constrained by a shape regex, which admits the space, so every
        natural tokenization (``" -17"``, ``"-17"``, ``" 17"``) stays available.
        Exact bounds are enforced by validation; ``generate`` resamples
        out-of-range draws (rejection sampling).
        """
        content = self._continuation_generate(prompt, numeric_regex(subschema), verbose=verbose)
        cast = int if subschema.get("type") == "integer" else float
        value = cast(content.strip())
        _check_json_schema_value(value, subschema, key="value")
        return value

    def _continuation_generate(
        self,
        prompt: str | tuple[str, str],
        schema: dict[str, Any] | str | list[str],
        verbose: bool = False,
    ) -> Any:
        if isinstance(schema, dict):
            structured_outputs_type = "json"
        elif isinstance(schema, list):
            structured_outputs_type = "choice"
        elif isinstance(schema, str):
            structured_outputs_type = "regex"
        else:
            raise ValueError(f"Invalid schema type: {type(schema)}")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {"structured_outputs": {structured_outputs_type: schema}},
        }

        # Base model
        if isinstance(prompt, str):
            kwargs["prompt"] = (
                f"{(self.system_prompt + '\n') if self.system_prompt else ''}{prompt}"
            )
            if verbose:
                print(f"Completion prompt: ```\n{kwargs['prompt']}\n```")
            response = self.client.completions.create(**kwargs)
            content = response.choices[0].text

        # Instruct model
        else:
            user, prefill = prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefill},
            ]
            if verbose:
                print(f"Chat prompt: ```\n{messages}\n```")
            kwargs["messages"] = messages
            # Merge prefill flags into the same extra_body that already holds structured_outputs.
            kwargs["extra_body"] = {
                **kwargs["extra_body"],
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        if verbose:
            print(f"Response: ```\n{content}\n```")

        if isinstance(schema, dict):
            return json.loads(content)
        return content
