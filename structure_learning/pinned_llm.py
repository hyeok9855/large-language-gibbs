from typing import Any

import numpy as np
from priorbot.priors import LLMPrior
from tqdm import tqdm


class PinnedLLMPrior(LLMPrior):
    """``LLMPrior`` that conditions by *pinning* observed values inside the
    generation schema rather than describing them in the prompt.

    When ``observed`` is provided, the observed features are added to the
    generation schema as single-value ``enum`` constraints (ordered first) and
    the model is asked to generate the *entire* data point as one JSON object.
    Grammar-constrained decoding then forces the observed fields to their
    values, while the remaining fields are sampled conditioned on them.

    This avoids the base-model failure mode where a *closed* observed JSON
    object is placed in the prompt and the model is then forced to emit a
    second, off-distribution JSON object for the resampled variables.

    The full per-feature schemas are passed via ``feature_schemas`` so that the
    pinned property keeps the feature's declared type (the per-step ``schema``
    only carries the resampled features). If a feature is missing from
    ``feature_schemas`` the type is simply omitted and inferred by the backend
    from the pinned ``enum`` value.
    """

    def __init__(self, *args: Any, feature_schemas: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.feature_schemas = feature_schemas or {}

    def _pin(self, key: str, value: Any) -> dict[str, Any]:
        pinned: dict[str, Any] = {}
        base_prop = self.feature_schemas.get(key)
        if base_prop is not None and "type" in base_prop:
            pinned["type"] = base_prop["type"]
        pinned["enum"] = [value]
        return pinned

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        observed = observed or {}
        samples = []
        for _ in tqdm(
            range(n_samples),
            disable=pbar is None,
            position=pbar,
            desc=f"Worker {pbar}",
            dynamic_ncols=True,
        ):
            resampled_keys = list(schema["required"])
            if self.shuffle_variables:
                np.random.shuffle(resampled_keys)

            gen_properties: dict[str, Any] = {}
            gen_required: list[str] = []
            # Observed features first so the resampled features are generated
            # (and thus conditioned) after them.
            for key, value in observed.items():
                gen_properties[key] = self._pin(key, value)
                gen_required.append(key)

            if self.manual_reasoning:
                gen_properties["reasoning"] = {
                    "type": "string",
                    "description": self.reasoning_prompt,
                }
                gen_required.append("reasoning")

            for key in resampled_keys:
                gen_properties[key] = schema["properties"][key]
                gen_required.append(key)

            gen_schema = {
                "type": "object",
                "properties": gen_properties,
                "required": gen_required,
            }

            prompt = self.template(gen_schema, observed)
            sample = self.llm.generate(prompt, gen_schema, verbose)

            if isinstance(sample, dict):
                sample.pop("reasoning", None)
                samples.append(sample)
            else:  # String should not be given as output (see .generate in llm.py)
                raise ValueError(f"LLM returned invalid output {sample}.")

        return samples
