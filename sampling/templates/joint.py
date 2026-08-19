"""Prompt templates for joint (non-iid) targets.

Wording follows the scalar templates, but a sample is one named vector
(X1, ..., Xd) rather than iid copies of a univariate law. JSON schemas are
built by ``Target.object_schema``.
"""

import json
from argparse import Namespace
from typing import Any, Callable

from sampling.targets import Target, get_target
from sampling.utils import indexed_var_names


def create_template(method: str, args: Namespace) -> Callable[..., str]:
    target: Target = get_target(args.target)
    if not target.is_joint:
        raise ValueError(f"Joint templates require a joint target, got {target.name!r}.")
    if method in ("indep", "batch"):
        raise ValueError(
            f"{method} sampling is a univariate baseline and is not defined for "
            f"joint target {target.name!r}."
        )

    model_type = args.model_type
    dist_str = target.describe(args)

    if method in ("gibbs", "direct"):
        if model_type == "base":

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return f"Here is a random sample from {dist_str}, formatted as JSON:\n"

                n_missing = len([key for key in schema["required"] if key != "reasoning"])
                remaining = (
                    "remaining coordinates of the same sample"
                    if n_missing > 1
                    else "remaining coordinate of the same sample"
                )
                return (
                    f"Here is a partial sample from {dist_str}:\n"
                    f"{json.dumps(target.format_observed(observed))}\n"
                    f"Here {'are' if n_missing > 1 else 'is'} the {remaining}, formatted as JSON:\n"
                )

        else:

            def template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
                if observed is None:
                    return (
                        f"Draw a random sample from {dist_str}. "
                        f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                    )

                n_missing = len([key for key in schema["required"] if key != "reasoning"])
                remaining = (
                    "remaining coordinates of the same sample"
                    if n_missing > 1
                    else "remaining coordinate of the same sample"
                )
                return (
                    f"You are sampling from {dist_str}. "
                    f"You have already observed these coordinates: "
                    f"{json.dumps(target.format_observed(observed))}. "
                    f"Draw the {remaining}. "
                    f"Respond with JSON that follows this schema: {json.dumps(schema)}"
                )

        return template

    if model_type != "instruct":
        raise ValueError("Barker or Gambling Gibbs only supports instruct model type")

    value_noun = "integers" if target.discrete else "values"

    if method == "barker_gibbs":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = f"You are sampling from {dist_str}. "
            if observed:
                _template += (
                    "You have already observed these coordinates: "
                    f"{json.dumps(target.format_observed(observed))}.\n"
                )
            option1_str = json.dumps(target.format_observed(option1))
            option2_str = json.dumps(target.format_observed(option2))
            _template += (
                "Which of the following two candidates for the unobserved coordinates "
                "is more likely under this joint distribution?\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}"
            )
            return _template

        return template

    if method == "gambling_gibbs":

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            output_schema: dict[str, Any],
            bet_value: float,
            observed: dict[str, Any] | None = None,
        ) -> str:
            _template = f"You are sampling from {dist_str}. "
            if observed:
                _template += (
                    "You have already observed these coordinates: "
                    f"{json.dumps(target.format_observed(observed))}.\n"
                )
            option1_str = json.dumps(target.format_observed(option1))
            option2_str = json.dumps(target.format_observed(option2))
            _template += (
                f"Consider two candidate {value_noun} for the unobserved coordinates:\n"
                f"Option 1: {option1_str}\n"
                f"Option 2: {option2_str}\n"
                "One of these is more plausible under the joint distribution than the other. "
                f"You may place a bet of ${bet_value} that Option 1 is more plausible "
                "than Option 2, which will pay out $100 if you are correct. "
                "Your aim is to maximise expected profit.\n"
                f"Respond with JSON that follows this schema: {json.dumps(output_schema)}"
            )
            return _template

        return template

    raise ValueError(f"Invalid method: {method}")


def create_continuation_template(args: Namespace) -> Callable[..., str | tuple[str, str]]:
    """Named JSON-object continuation for direct_continuation and gibbs_continuation."""
    target: Target = get_target(args.target)
    if not target.is_joint:
        raise ValueError(f"Joint continuation requires a joint target, got {target.name!r}.")

    dist_str = target.describe(args)
    var_names = indexed_var_names(args.gibbs_k_vars)
    if len(var_names) > 2:
        var_str = f"{var_names[0]}, ..., {var_names[-1]}"
    else:
        var_str = ", ".join(var_names)
    fmt = target.value_formatter

    def prefill(observed: dict[str, Any] | None, next_key: str) -> str:
        items: list[str] = []
        if observed:
            items.extend(
                f"{json.dumps(key)}: {json.dumps(fmt(value))}" for key, value in observed.items()
            )
        items.append(f"{json.dumps(next_key)}:")
        return "{" + ", ".join(items)

    if args.model_type == "base":

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            if not next_key:
                raise ValueError("Joint continuation requires next_key.")
            return (
                f"Here is a random sample of ({var_str}) from {dist_str}, formatted as JSON:\n"
                + prefill(observed, next_key)
            )

    else:

        def template(
            observed: dict[str, Any] | None = None, next_key: str | None = None
        ) -> str | tuple[str, str]:
            if not next_key:
                raise ValueError("Joint continuation requires next_key.")
            return (
                f"Draw a random sample of ({var_str}) from {dist_str}.",
                prefill(observed, next_key),
            )

    return template
