import json
from argparse import Namespace
from typing import Any, Callable

from sampling.utils import round_dict


def create_template_and_schema(
    method: str, args: Namespace
) -> tuple[Callable[..., str], dict[str, Any]]:
    model_type = args.model_type
    mean = args.mean
    std = args.std
    mcmc_sigma_multiplier = args.mcmc_sigma_multiplier
    assert std > 0 and mcmc_sigma_multiplier > 0

    # Independent Sampling
    if method == "indep":
        if model_type == "base":

            def template(schema: dict[str, Any], observed=None) -> str:
                return f"Here is a random sample from a Gaussian distribution with mean {mean} and standard deviation {std}, formatted as JSON:\n"

        else:

            def template(schema: dict[str, Any], observed=None) -> str:
                return (
                    f"Draw a random sample from a Gaussian distribution with mean {mean} and standard deviation {std}.\n"
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

        schema = {
            "type": "object",
            "properties": {
                "sample": {
                    "type": "number",
                    "minimum": mean - mcmc_sigma_multiplier * std,
                    "maximum": mean + mcmc_sigma_multiplier * std,
                }
            },
            "required": ["sample"],
        }

        return template, schema

    # Batch Sampling
    if method == "batch":
        n_samples_per_chain = args.n_samples_per_chain

        if model_type == "base":

            def template(schema: dict[str, Any], observed=None) -> str:
                return f"Here are {n_samples_per_chain} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}, formatted as JSON:\n"

        else:

            def template(schema: dict[str, Any], observed=None) -> str:
                return (
                    f"Draw {n_samples_per_chain} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}.\n"
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

        schema = {
            "type": "object",
            "properties": {
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "minimum": mean - mcmc_sigma_multiplier * std,
                        "maximum": mean + mcmc_sigma_multiplier * std,
                    },
                    "minItems": n_samples_per_chain,
                    "maxItems": n_samples_per_chain,
                }
            },
            "required": ["samples"],
        }

        return template, schema

    # Gibbs, Barker Gibbs, Gambling Gibbs Sampling; they share the same schema
    k_vars = args.gibbs_k_vars
    schema = {
        "type": "object",
        "properties": {
            f"X{i}": {
                "type": "number",
                "minimum": mean - mcmc_sigma_multiplier * std,
                "maximum": mean + mcmc_sigma_multiplier * std,
            }
            for i in range(k_vars)
        },
        "required": [f"X{i}" for i in range(k_vars)],
    }

    # Gibbs Sampling
    if method == "gibbs":

        if model_type == "base":

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return f"Here are {k_vars} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}, formatted as JSON:\n"

                n_missing = len(schema["properties"])
                _template = (
                    f"Here are {len(observed)} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}:\n"
                    f"{json.dumps(round_dict(observed))}\n"
                )
                _template += (
                    f"Here is another set of {n_missing} iid samples from the same distribution, formatted as JSON:\n"
                    if n_missing > 1
                    else "Here is another random sample from the same distribution, formatted as JSON:\n"
                )
                return _template

        else:

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return (
                        f"Draw {k_vars} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}.\n"
                        f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                    )

                n_missing = len(schema["properties"])
                _template = (
                    f"You are generating {k_vars} iid samples from a Gaussian distribution with mean {mean} and standard deviation {std}.\n"
                    f"You have already observed {len(observed)} iid samples: {json.dumps(round_dict(observed))}.\n"
                )
                _template += (
                    f"Draw another set of {n_missing} iid random samples from the same distribution. Respond with JSON that follows this schema: {json.dumps(schema)}\n"
                    if n_missing > 1
                    else f"Draw another random sample from the same distribution. Respond with JSON that follows this schema: {json.dumps(schema)}\n"
                )
                return _template

        return template, schema

    assert (
        args.model_type == "instruct"
    ), "Barker or Gambling Gibbs only supports instruct model type"

    # Barker Gibbs Sampling
    if method == "barker":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = (
                f"You are generating iid samples from a Gaussian distribution with mean {mean} "
                f"and standard deviation {std}.\n"
            )
            if observed:
                _template += f"You have already observed: {json.dumps(round_dict(observed))}.\n"
            option1_str = json.dumps(round_dict(option1))
            option2_str = json.dumps(round_dict(option2))
            _template += (
                "Which of the following two candidates is more likely to be the iid sample from the distribution?\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}."
            )
            return _template

        return template, schema

    # Gambling Gibbs Sampling
    if method == "gambling":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            bet_value: float,
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = (
                f"You are generating iid samples from a Gaussian distribution with mean {mean} "
                f"and standard deviation {std}.\n"
            )
            if observed:
                _template += f"You have already observed: {json.dumps(round_dict(observed))}.\n"
            option1_str = json.dumps(round_dict(option1))
            option2_str = json.dumps(round_dict(option2))
            _template += (
                "Consider two candidate values for the next iid sample:\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                "One of these is more plausible under the distribution than the other. "
                f"You may place a bet of ${bet_value} that Option 1 is more plausible than Option 2, "
                "which will pay out $100 if you are correct. Your aim is to maximise expected profit.\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}."
            )
            return _template

        return template, schema

    raise ValueError(f"Invalid method: {method}")
