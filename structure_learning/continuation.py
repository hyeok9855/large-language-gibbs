import json
from collections.abc import Callable
from typing import Any

import numpy as np
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import AsyncPrior
from tqdm import tqdm

MAX_VALUE_TOKENS = 32

# Narrower than re.escape, which also escapes space, "-", and "&".
_REGEX_METACHARS = r"\^$.|?*+()[]{}"


def _escape_literal(value: str) -> str:
    if json.dumps(value)[1:-1] != value:
        raise ValueError(
            f"Enum value {value!r} needs JSON escaping, which the shape regex does not model."
        )
    return "".join(f"\\{char}" if char in _REGEX_METACHARS else char for char in value)


def quoted_choice_regex(choices: list[str]) -> str:
    """Optional space, then one quoted enum member.

    Not vLLM's ``structured_outputs.choice``: that matches literals
    byte-for-byte and hides the leading-space token.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    alternatives = "|".join(_escape_literal(choice) for choice in choices)
    return rf' ?"({alternatives})"'


def partial_json(observed: dict[str, Any] | None, next_key: str) -> str:
    """Unclosed JSON object ending at ``"<next_key>":``."""
    items = [f"{json.dumps(key)}: {json.dumps(value)}" for key, value in (observed or {}).items()]
    items.append(f"{json.dumps(next_key)}:")
    return "{" + ", ".join(items)


class ContinuationOpenAICompatLLM(OpenAICompatLLM):
    def generate(
        self,
        prompt: str | tuple[str, str],
        subschema: dict[str, Any],
        verbose: bool = False,
        max_trials: int = 10,
    ) -> str:
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1.")

        choices = subschema.get("enum")
        if subschema.get("type") != "string" or not choices:
            raise ValueError(f"Continuation sampling supports string enums only; got {subschema}")

        for i in range(max_trials):
            try:
                return self._generate_choice(prompt, list(choices), verbose=verbose)
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
        content = self._continuation_generate(prompt, quoted_choice_regex(choices), verbose)
        try:
            value = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Response {content!r} is not a JSON string") from exc
        if value not in choices:
            raise ValueError(f"Response {value!r} is not in choice set {choices}")
        return value

    def _continuation_generate(
        self,
        prompt: str | tuple[str, str],
        regex: str,
        verbose: bool = False,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": min(self.max_tokens, MAX_VALUE_TOKENS),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": {"structured_outputs": {"regex": regex}},
        }

        if isinstance(prompt, str):
            kwargs["prompt"] = (
                f"{(self.system_prompt + '\n') if self.system_prompt else ''}{prompt}"
            )
            if verbose:
                print(f"Completion prompt: ```\n{kwargs['prompt']}\n```")
            response = self.client.completions.create(**kwargs)
            content = response.choices[0].text
        else:
            user, prefill = prompt
            kwargs["messages"] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": prefill},
            ]
            kwargs["extra_body"] = {
                **kwargs["extra_body"],
                "add_generation_prompt": False,
                "continue_final_message": True,
            }
            if verbose:
                print(f"Chat prompt: ```\n{kwargs['messages']}\n```")
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        content = content or ""
        if verbose:
            print(f"Response: ```\n{content}\n```")
        return content


class ContinuationLLMPrior(AsyncPrior):
    def __init__(
        self,
        llm: ContinuationOpenAICompatLLM,
        template: Callable[..., str | tuple[str, str]],
        shuffle_variables: bool = True,
    ):
        self.llm = llm
        self.template = template
        self.shuffle_variables = shuffle_variables

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        samples = []
        for _ in tqdm(
            range(n_samples),
            disable=pbar is None,
            position=pbar,
            desc=f"Worker {pbar}",
            dynamic_ncols=True,
        ):
            keys = list(schema["required"])
            if self.shuffle_variables:
                np.random.shuffle(keys)

            context = dict(observed or {})
            key_order = list(context.keys()) + keys
            result: dict[str, Any] = {}
            for key in keys:
                prompt = self.template(context or None, next_key=key, key_order=key_order)
                value = self.llm.generate(prompt, schema["properties"][key], verbose)
                result[key] = value
                context[key] = value

            samples.append(result)

        return samples
