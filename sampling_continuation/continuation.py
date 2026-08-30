"""Prefill continuation without a JSON grammar: numeric field values and choice
answers, drawn by continuing text already in the context.

Design rule for every pattern here: admit an optional leading space, so each
natural tokenization stays available (" -17" as well as "-17"). Constraining it
away forces a boundary BPE never produces and silently distorts the samples - a
bounded JSON-number grammar after a committed prefix gave 0% negatives on
symmetric supports. Bounds are enforced by validate-and-retry instead.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
from priorbot.llm import OpenAICompatLLM, _check_json_schema_value
from priorbot.priors import AsyncPrior, BarkerGibbsLLMPrior, GamblingGibbsLLMPrior
from tqdm import tqdm

from sampling.reasoning_traces import TraceRecorder

MAX_FRACTION_DIGITS = 6

REASONING_PREFIX = "Reasoning:"
ANSWER_PREFIX = "Answer:"
REASONING_STOP = ["Answer:", "answer:"]


def numeric_regex(subschema: dict[str, Any]) -> str:
    """Optional space, optional minus, digit budget sized from the bounds."""
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


def choice_regex(choices: list[str]) -> str:
    """Optional space, then one choice verbatim.

    Not vLLM's ``structured_outputs.choice``: that matches the literals
    byte-for-byte, masking the leading-space token 100% of calls took.
    """
    if not choices:
        raise ValueError("choices must be non-empty")
    # re.escape escapes spaces too; keep them plain for the grammar compiler.
    alternatives = "|".join(re.escape(choice).replace("\\ ", " ") for choice in choices)
    return f" ?({alternatives})"


def _extend_prompt(prompt: str | tuple[str, str], suffix: str) -> str | tuple[str, str]:
    """Append at the continuation point: the instruct prefill, or the base prompt."""
    if isinstance(prompt, tuple):
        user, prefill = prompt
        return user, prefill + suffix
    return prompt + suffix


class ContinuationOpenAICompatLLM(OpenAICompatLLM):
    """LLM client for prefill continuation on base or instruct models."""

    #: Output budget for ``generate_choice``'s free-text reasoning pass.
    reasoning_max_tokens: int = 1024

    #: Chat-template renderer kwargs, e.g. ``{"enable_thinking": False}``. Some
    #: templates treat an *undefined* thinking flag as enabled and then inject
    #: reasoning instructions into the system message, altering the prompt.
    chat_template_kwargs: dict[str, Any] | None = None

    #: Extra keys on the assistant prefill message. gemma-4 emits its mandatory
    #: thought-channel block only for a message carrying ``reasoning``, and the
    #: markers cannot go in ``content``, which the template strips.
    prefill_extra: dict[str, Any] | None = None

    def generate(
        self,
        prompt: str | tuple[str, str],
        subschema: dict[str, Any],
        verbose: bool = False,
        max_trials: int = 10,
    ) -> Any:
        """One numeric field value, continuing a partial JSON array/object.

        The shape regex admits much more than the schema window, so
        out-of-range draws are retried; keep ``max_trials`` generous or a model
        whose mass sits partly outside the window loses the whole result file.
        """
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1.")
        if subschema.get("type") not in ("integer", "number"):
            raise ValueError(f"Unsupported subschema for continuation generation: {subschema}")

        for i in range(max_trials):
            try:
                content = self._continuation_generate(
                    prompt, numeric_regex(subschema), verbose=verbose
                )
                cast = int if subschema["type"] == "integer" else float
                value = cast(content.strip())
                _check_json_schema_value(value, subschema, key="value")
                return value
            except Exception as exc:
                print(f"Error during field value generation:\n{exc}")
                if i < max_trials - 1:
                    print(f"Retrying ({i + 1}/{max_trials}) ...")

        raise RuntimeError(f"Failed to generate a valid field value after {max_trials} trials.")

    def generate_choice(
        self,
        prompt: str | tuple[str, str],
        choices: list[str],
        manual_reasoning: bool = False,
        verbose: bool = False,
        max_trials: int = 10,
        return_reasoning: bool = False,
    ) -> str | tuple[str, str | None]:
        """Answer a choice question, optionally after a free-text reasoning pass.

        Plain: one constrained call on a prefill ending "Answer:". With
        ``manual_reasoning``: an unconstrained "Reasoning:" pass first (stopping
        at "Answer:" or the cap; truncation used as-is), then a constrained call
        whose prefill replays the trace, making the choice a marginal over
        sampled traces. A failed trial resamples the trace too, so (trace,
        answer) stay jointly sampled. Base-model prompts must end with a newline
        so the marker starts a clean line.
        """
        if max_trials < 1:
            raise ValueError("max_trials must be at least 1.")

        for i in range(max_trials):
            try:
                reasoning = None
                if manual_reasoning:
                    reasoning = self._continuation_generate(
                        _extend_prompt(prompt, REASONING_PREFIX),
                        regex=None,
                        verbose=verbose,
                        stop=REASONING_STOP,
                        max_tokens=self.reasoning_max_tokens,
                    )
                    answer_prefix = f"{REASONING_PREFIX}{reasoning.rstrip()}\n{ANSWER_PREFIX}"
                else:
                    answer_prefix = ANSWER_PREFIX
                content = self._continuation_generate(
                    _extend_prompt(prompt, answer_prefix), choice_regex(choices), verbose=verbose
                ).strip()
                if content not in choices:
                    raise ValueError(f"Response {content!r} is not in choice set {choices}")
                return (content, reasoning) if return_reasoning else content
            except Exception as exc:
                print(f"Error during choice generation:\n{exc}")
                if i < max_trials - 1:
                    print(f"Retrying ({i + 1}/{max_trials}) ...")

        raise RuntimeError(f"Failed to generate a valid choice after {max_trials} trials.")

    def _continuation_generate(
        self,
        prompt: str | tuple[str, str],
        regex: str | None,
        verbose: bool = False,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One raw call. ``regex=None`` runs unconstrained (the reasoning pass);
        vLLM strips ``stop`` strings from the returned text."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if regex is not None:
            kwargs["extra_body"] = {"structured_outputs": {"regex": regex}}
        if stop is not None:
            kwargs["stop"] = stop

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
                {"role": "assistant", "content": prefill, **(self.prefill_extra or {})},
            ]
            if verbose:
                print(f"Chat prompt: ```\n{messages}\n```")
            kwargs["messages"] = messages
            # extra_body may already hold structured_outputs.
            kwargs["extra_body"] = {
                **kwargs.get("extra_body", {}),
                "add_generation_prompt": False,
                "continue_final_message": True,
                **(
                    {"chat_template_kwargs": self.chat_template_kwargs}
                    if self.chat_template_kwargs
                    else {}
                ),
            }
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content

        content = content or ""
        if verbose:
            print(f"Response: ```\n{content}\n```")
        return content


class ContinuationLLMPrior(AsyncPrior):
    """Field-by-field prefill continuation - the continuation counterpart of
    priorbot's ``LLMPrior``. Fills any ``Target.object_schema``: scalar keys one
    call each, a fixed-length array key one call per element. No
    ``manual_reasoning`` variant: prose inside the prefill would break the frame."""

    #: Rejection-sampling budget per field; see ``generate``.
    max_trials: int = 100

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
            # Chains run in threads; unlike priorbot's LLMPrior, never mutate
            # the caller's schema.
            gen_schema = deepcopy(schema)
            if self.shuffle_variables:
                keys = list(gen_schema["properties"].keys())
                np.random.shuffle(keys)
                gen_schema["properties"] = {key: gen_schema["properties"][key] for key in keys}
                gen_schema["required"] = keys

            context = dict(observed or {})
            result: dict[str, Any] = {}

            def draw(field_key: str, field_schema: dict[str, Any]) -> Any:
                value = self.llm.generate(
                    self.template(context or None, next_key=field_key),
                    field_schema,
                    verbose=verbose,
                    max_trials=self.max_trials,
                )
                context[field_key] = value
                return value

            for key in gen_schema["required"]:
                subschema = gen_schema["properties"][key]
                if subschema.get("type") == "array":
                    # The JSON family's `batch` shape: one call per element,
                    # positionally keyed so the prefill shows every value drawn
                    # so far. minItems == maxItems, set by Target.object_schema.
                    result[key] = [
                        draw(f"{key}[{i}]", subschema["items"])
                        for i in range(subschema["minItems"])
                    ]
                else:
                    result[key] = draw(key, subschema)

            samples.append(result)

        return samples


class ContinuationBarkerGibbsLLMPrior(BarkerGibbsLLMPrior):
    """Barker-Gibbs with a choice-regex acceptance step instead of a JSON object;
    proposal and block machinery inherited. ``template(option1, option2,
    observed)`` returns the user message only - the client adds the markers."""

    CHOICES = ["Option 1", "Option 2"]

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
        assert isinstance(self.llm, ContinuationOpenAICompatLLM)
        answer = self.llm.generate_choice(
            self.template(option1, option2, observed),
            self.CHOICES,
            manual_reasoning=self.manual_reasoning,
            verbose=verbose,
        )
        return answer == self.CHOICES[0]


class ContinuationGamblingGibbsLLMPrior(GamblingGibbsLLMPrior):
    """As ``ContinuationBarkerGibbsLLMPrior``, with the bet question.
    ``template(option1, option2, bet_value, observed)``."""

    CHOICES = ["Place Bet", "Do Not Place Bet"]

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
        assert isinstance(self.llm, ContinuationOpenAICompatLLM)
        # Random stake, as in priorbot's GamblingLLMPrior._acceptance.
        bet_value = np.round(np.random.rand() * 100, 2)
        answer = self.llm.generate_choice(
            self.template(option1, option2, bet_value, observed),
            self.CHOICES,
            manual_reasoning=self.manual_reasoning,
            verbose=verbose,
        )
        return answer == self.CHOICES[0]


def install_choice_recorder(llm, recorder: TraceRecorder) -> None:
    """Snapshot the first ``recorder.n_traces`` acceptance calls' reasoning.

    ``install_json`` reads it off the returned dict; the two-call protocol
    yields it separately, so ask for it and return only the answer.
    """
    inner = llm.generate_choice

    def wrapper(prompt, choices, manual_reasoning=False, **kwargs):
        index = recorder.claim()
        if index is None:
            return inner(prompt, choices, manual_reasoning=manual_reasoning, **kwargs)
        kwargs.pop("return_reasoning", None)
        answer, reasoning = inner(
            prompt, choices, manual_reasoning=manual_reasoning, return_reasoning=True, **kwargs
        )
        recorder.record(prompt, reasoning, answer, index)
        return answer

    llm.generate_choice = wrapper
