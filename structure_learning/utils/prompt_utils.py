"""Shared utilities for building LLM prompts about a Bayesian network dataset.

Both `generate_prior_data.py` and `generate_prior_matrix.py` build prompts
from the same `meta_data.json` schema, so the helpers below provide a single
canonical way to format the system prompt, dataset description, and feature
description block.
"""

from pathlib import Path
from typing import Any
import json

import numpy as np


MODEL_NAME_TO_TYPE: dict[str, str] = {
    "meta-llama/Llama-3.1-8B": "base",
    "meta-llama/Llama-3.1-8B-Instruct": "instruct",
    "meta-llama/Llama-3.1-70B": "base",
    "meta-llama/Llama-3.1-70B-Instruct": "instruct",
    "allenai/Olmo-3-1125-32B": "base",
    "allenai/Olmo-3-32B-Think": "instruct",
}


def load_meta(path: Path | str) -> dict:
    """Load a dataset's `meta_data.json`."""
    with open(path) as f:
        return json.load(f)


def build_system_prompt(meta: dict, task_description: str) -> str:
    """Build the system prompt for an instruction-tuned LLM.

    Args:
        meta: Parsed `meta_data.json` for the dataset.
        task_description: Verb phrase describing the LLM's task, e.g.
            ``"generating realistic data points"`` or
            ``"discovering the structure of a Bayesian network"``.
    """
    field = meta["field"]
    return (
        f"You are a data scientist and expert in the field of {field}, "
        f"tasked with {task_description} for a given dataset description."
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
