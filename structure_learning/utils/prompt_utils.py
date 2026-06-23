"""Shared utilities for building LLM prompts about a Bayesian network dataset.

Both `generate_prior_data.py` and `generate_prior_matrix.py` build prompts
from the same `meta_data.json` schema, so the helpers below provide a single
canonical way to format the system prompt, dataset description, and feature
description block.
"""


def build_system_prompt(meta: dict) -> str:
    """Build the system prompt for an instruction-tuned LLM.

    Args:
        meta: Parsed `meta_data.json` for the dataset.
    """
    field = meta["field"]
    return (
        f"You are a data scientist and an expert in {field}, "
        f"tasked with generating realistic data points for a given dataset description."
    )


def get_dataset_description(meta: dict) -> str:
    """Format the dataset description block."""
    return f"[Dataset description] {meta['dataset_description']}"


def get_feature_description(
    meta: dict,
    observed_keys: list[str],
    unobserved_keys: list[str],
) -> str:
    """Format the feature description block.

    Args:
        meta: Parsed `meta_data.json` for the dataset.
        observed_keys: List of observed feature names.
        unobserved_keys: List of unobserved feature names.

    Returns:
        A string ``[Feature description] {feature_description}``.
    """
    feature_description = ", ".join(
        f'"{name}": {meta["features"][name]["description"]}'
        for name in observed_keys + unobserved_keys
    )
    return f"[Feature description] {feature_description}."
