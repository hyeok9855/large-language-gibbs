"""Prompt templates for scalar (univariate iid) targets.

The wording is shared across every distribution; only the description of the
target and how values are rendered differ, and those come from the ``Target``
registry in ``sampling.targets``. JSON schemas are built by ``Target.object_schema``.
"""

import json
from argparse import Namespace
from typing import Any, Callable

from sampling.targets import Target, get_target


def create_template(method: str, args: Namespace) -> Callable[..., str]:
    target: Target = get_target(args.target)
    if target.is_joint:
        raise ValueError(f"Scalar templates require a univariate target, got {target.name!r}.")

    model_type = args.model_type
    dist_str = target.describe(args)

    # Independent Sampling
    if method == "indep":
        if model_type == "base":

            def template(schema: dict[str, Any], observed=None) -> str:
                return f"Here is a random sample from {dist_str}, formatted as JSON:\n"

        else:

            def template(schema: dict[str, Any], observed=None) -> str:
                return (
                    f"Draw a random sample from {dist_str}. "
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

        return template

    # Batch Sampling
    if method == "batch":
        n_samples_per_chain = args.n_samples_per_chain

        if model_type == "base":

            def template(schema: dict[str, Any], observed=None) -> str:
                return (
                    f"Here are {n_samples_per_chain} iid samples from {dist_str}, "
                    f"formatted as JSON:\n"
                )

        else:

            def template(schema: dict[str, Any], observed=None) -> str:
                return (
                    f"Draw {n_samples_per_chain} iid samples from {dist_str}. "
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

        return template

    k_vars = args.gibbs_k_vars

    # Gibbs and direct sampling share the same joint-variable template
    if method in ("gibbs", "direct"):
        if model_type == "base":

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return f"Here are {k_vars} iid samples from {dist_str}, formatted as JSON:\n"

                n_missing = len([key for key in schema["required"] if key != "reasoning"])
                _template = (
                    f"Here are {len(observed)} iid samples from {dist_str}:\n"
                    f"{json.dumps(target.format_observed(observed))}\n"
                )
                if n_missing > 1:
                    _template += (
                        f"Here is another set of {n_missing} iid samples from the same "
                        "distribution, formatted as JSON:\n"
                    )
                else:
                    _template += (
                        "Here is another random sample from the same distribution, "
                        "formatted as JSON:\n"
                    )
                return _template

        else:

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return (
                        f"Draw {k_vars} iid samples from {dist_str}. "
                        f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                    )

                n_missing = len([key for key in schema["required"] if key != "reasoning"])
                _template = (
                    f"You are generating {k_vars} iid samples from {dist_str}. "
                    f"You have already observed {len(observed)} iid samples: "
                    f"{json.dumps(target.format_observed(observed))}\n"
                )
                if n_missing > 1:
                    _template += (
                        f"Draw another set of {n_missing} iid samples from the same "
                        "distribution. "
                        f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                    )
                else:
                    _template += (
                        "Draw another random sample from the same distribution. "
                        f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                    )
                return _template

        return template

    if model_type != "instruct":
        raise ValueError("Barker or Gambling Gibbs only supports instruct model type")

    # Barker Gibbs Sampling
    if method == "barker_gibbs":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = f"You are generating {k_vars} iid samples from {dist_str}. "
            if observed:
                _template += (
                    f"You have already observed {len(observed)} iid samples: "
                    f"{json.dumps(target.format_observed(observed))}\n"
                )
            option1_str = json.dumps(target.format_observed(option1))
            option2_str = json.dumps(target.format_observed(option2))
            _template += (
                "Which of the following two candidates is more likely to be the remaining iid "
                "samples from the distribution?\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}"
            )
            return _template

        return template

    # Gambling Gibbs Sampling
    if method == "gambling_gibbs":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            bet_value: float,
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = f"You are generating {k_vars} iid samples from {dist_str}. "
            if observed:
                _template += (
                    f"You have already observed {len(observed)} iid samples: "
                    f"{json.dumps(target.format_observed(observed))}\n"
                )
            option1_str = json.dumps(target.format_observed(option1))
            option2_str = json.dumps(target.format_observed(option2))
            _template += (
                "Consider two candidates for the remaining iid samples:\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                "One of these is more plausible under the distribution than the other. "
                f"You may place a bet of ${bet_value} that Option 1 is more plausible "
                "than Option 2, which will pay out $100 if you are correct. "
                "Your aim is to maximise expected profit.\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}"
            )
            return _template

        return template

    raise ValueError(f"Invalid method: {method}")


def create_continuation_template(args: Namespace) -> Callable[..., str | tuple[str, str]]:
    """Field-by-field JSON-array continuation for direct_continuation and gibbs_continuation."""
    target: Target = get_target(args.target)
    if target.is_joint:
        raise ValueError(f"Scalar continuation requires a univariate target, got {target.name!r}.")

    dist_str = target.describe(args)
    num_vars = args.gibbs_k_vars
    fmt = target.value_formatter

    def prefill(observed: dict[str, Any] | None) -> str:
        if observed is None:
            return "["
        return "[" + ", ".join(json.dumps(fmt(v)) for v in observed.values()) + ","

    if args.model_type == "base":

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            return f"Here are {num_vars} iid samples from {dist_str}:\n" + prefill(observed)

    else:

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            return f"Draw {num_vars} iid samples from {dist_str}.", prefill(observed)

    return template
