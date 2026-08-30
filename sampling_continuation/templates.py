import json
from argparse import Namespace
from typing import Any, Callable

from sampling.targets import Target, get_target
from sampling.templates import round_dict


def array_prefill(observed: dict[str, Any] | None) -> str:
    """Open the array and replay what is already drawn. Ends dry on "[" or ","
    and never on nothing: numeric_regex owns the optional space."""
    if observed is None:
        return "["
    values = round_dict(observed).values()
    return "[" + ", ".join(json.dumps(v) for v in values) + ","


def values_str(values: list[Any]) -> str:
    """Bare list, matching the field methods' array prefill. Never a
    {"X1": ...} object: these are exchangeable slots, not variables, and naming
    them invents an identity and reveals which slot is in play - models were
    seen narrating "the 16th sample" while judging X3."""
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def option_str(option: dict[str, Any], keys: list[str]) -> str:
    """One candidate; ``keys`` is shared by both options so a multi-site block
    stays aligned position-by-position."""
    formatted = round_dict(option, 2)
    if len(keys) == 1:
        return json.dumps(formatted[keys[0]])
    return values_str([formatted[key] for key in keys])


def options_block(option1: dict[str, Any], option2: dict[str, Any], keys: list[str]) -> str:
    return f"Option 1: {option_str(option1, keys)}\nOption 2: {option_str(option2, keys)}\n"


def candidate_preamble(k_vars: int, dist_str: str, observed: dict[str, Any] | None) -> str:
    """Shared opening of both decision kernels: the budget, then the draws so far.

    Order comes from the caller: MetropolisWithinGibbsLLMPrior presents ``observed``
    in a fresh random order each iteration, so position carries no information."""
    _template = f"You are generating {k_vars} iid samples from {dist_str}. "
    if observed:
        values = list(round_dict(observed, 2).values())
        _template += f"You have already drawn {len(values)} iid samples:\n"
        _template += f"{values_str(values)}\n"
    return _template


def choice_format_instruction(choices: list[str], manual_reasoning: bool) -> str:
    """Format line for the decision kernels."""
    quoted = " or ".join(f"'{choice}'" for choice in choices)
    if manual_reasoning:
        return (
            "First write your reasoning after 'Reasoning:', then give your final "
            f"answer after 'Answer:' as exactly {quoted}."
        )
    return f"Answer with exactly {quoted}."


def create_template(args: Namespace, method: str) -> Callable[..., str | tuple[str, str]]:
    """Continuation template for ``method``; the schema comes from
    ``Target.object_schema`` but is never rendered.

    Mirrors ``sampling.templates.create_template`` block for block. Field
    methods return ``template(observed, next_key)`` and the decision kernels
    ``template(option1, option2, [bet_value,] observed)`` - neither takes a
    schema argument, so the returned prompt is a ``(user, prefill)`` pair for
    instruct models and a single continuation string for base models.
    """
    target: Target = get_target(args.target)
    model_type = args.model_type
    dist_str = target.description(args)

    # Independent Sampling
    if method == "indep":
        if model_type == "base":

            def template(observed=None, next_key=None) -> str:
                return f"Here is a random sample from {dist_str}:\n"

        else:

            def template(observed=None, next_key=None) -> tuple[str, str]:
                return f"Draw a random sample from {dist_str}. Answer with a number.", ""

        return template

    # Batch Sampling
    if method == "batch":
        n_samples_per_chain = args.n_samples_per_chain

        if model_type == "base":

            def template(observed=None, next_key=None) -> str:
                return (
                    f"Here are {n_samples_per_chain} iid samples from {dist_str}:\n"
                    + array_prefill(observed)
                )

        else:

            def template(observed=None, next_key=None) -> tuple[str, str]:
                return (
                    f"Draw {n_samples_per_chain} iid samples from {dist_str}. "
                    f"Answer with a list of {n_samples_per_chain} numbers."
                ), array_prefill(observed)

        return template

    k_vars = args.gibbs_k_vars

    # Gibbs Sampling
    if method == "gibbs":
        if model_type == "base":

            def template(
                observed: dict[str, Any] | None = None, next_key: str | None = None
            ) -> str:
                return f"Here are {k_vars} iid samples from {dist_str}:\n" + array_prefill(observed)

        else:

            def template(
                observed: dict[str, Any] | None = None, next_key: str | None = None
            ) -> tuple[str, str]:
                return (
                    f"Draw {k_vars} iid samples from {dist_str}. "
                    f"Answer with a list of {k_vars} numbers."
                ), array_prefill(observed)

        return template

    if model_type != "instruct":
        raise ValueError("Barker or Gambling Gibbs only supports instruct model type")

    # Barker Gibbs Sampling
    if method == "barker_gibbs":
        instruction = choice_format_instruction(["Option 1", "Option 2"], args.manual_reasoning)

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            observed: dict[str, Any] | None = None,
        ) -> tuple[str, str]:
            keys = list(option1.keys())
            noun = "sample" if len(keys) == 1 else "samples"
            user = (
                candidate_preamble(k_vars, dist_str, observed)
                + "Which of the following two candidates is more likely to be the "
                f"next iid {noun} from the distribution?\n"
                + options_block(option1, option2, keys)
                + instruction
            )
            return user, ""

        return template

    # Gambling Gibbs Sampling
    if method == "gambling_gibbs":
        instruction = choice_format_instruction(
            ["Place Bet", "Do Not Place Bet"], args.manual_reasoning
        )

        def template(
            option1: dict[str, Any],
            option2: dict[str, Any],
            bet_value: float,
            observed: dict[str, Any] | None = None,
        ) -> tuple[str, str]:
            keys = list(option1.keys())
            noun = "sample" if len(keys) == 1 else "samples"
            user = (
                candidate_preamble(k_vars, dist_str, observed)
                + f"Consider two candidates for the next iid {noun}:\n"
                + options_block(option1, option2, keys)
                + "One of these is more plausible under the distribution than the other. "
                f"You may place a bet of ${bet_value} that Option 1 is more plausible "
                "than Option 2, which will pay out $100 if you are correct. "
                "Your aim is to maximise expected profit.\n" + instruction
            )
            return user, ""

        return template

    raise ValueError(f"Invalid method: {method}")
