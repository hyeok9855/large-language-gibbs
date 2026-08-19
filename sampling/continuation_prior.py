"""Field-by-field conditional sampling via manual continuation."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
from priorbot.priors import AsyncPrior
from tqdm import tqdm

from sampling.continuation_llm import ContinuationOpenAICompatLLM


class ContinuationLLMPrior(AsyncPrior):
    """Sample object fields one at a time with manual continuation."""

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
            gen_schema = deepcopy(schema)
            if self.shuffle_variables:
                keys = list(gen_schema["properties"].keys())
                np.random.shuffle(keys)
                gen_schema["properties"] = {key: gen_schema["properties"][key] for key in keys}
                gen_schema["required"] = keys

            context = dict(observed or {})
            result: dict[str, Any] = {}

            for key in gen_schema["required"]:
                subschema = gen_schema["properties"][key]
                prompt = self.template(context or None, next_key=key)
                value = self.llm.generate(prompt, subschema, verbose=verbose)
                result[key] = value
                context[key] = value

            samples.append(result)

        return samples
