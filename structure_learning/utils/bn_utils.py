import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.sampling import BayesianModelSampling

# Dataset-specific fix-ups shared by dataset creation (get_pgmpy_dataset.py),
# LLM data generation (generate_llm_data.py) and scoring (utils/score_utils.py).
# Dataset names are accepted in both "bnrep/xxx" and "bnrep_xxx" forms.
#
# STATE_RENAMES: raw pgmpy state name -> state name used in the saved data
# CSVs, meta_data.json and cpds.pkl.
STATE_RENAMES = {
    "bnrep_algalactivity1": {"0": "low", "1": "high"},
    "bnrep_algalactivity2": {"0": "low", "1": "high"},
    # The bnRep model misspells the "Female" state of Gender as "Femal".
    "bnrep_tubercolosis": {"Femal": "Female"},
}

# FEATURE_RENAMES: raw pgmpy node name (used as data CSV column) -> feature
# name shown to the LLM in meta_data.json (e.g., to fix typos). LLM-generated
# data is renamed back to the raw node names before saving.
FEATURE_RENAMES = {
    "bnrep_tubercolosis": {"Tubercolosis": "Tuberculosis"},
    "bnrep_knowledge": {"C": "C#"},
}


def get_state_renames(dataset_name: str) -> dict[str, str]:
    return STATE_RENAMES.get(dataset_name.replace("/", "_"), {})


def get_feature_renames(dataset_name: str) -> dict[str, str]:
    return FEATURE_RENAMES.get(dataset_name.replace("/", "_"), {})


def get_true_domains(model, dataset_name: str) -> dict[str, list]:
    """Full per-variable domains of a ground-truth pgmpy model, with the
    dataset-specific state renames applied so that they match the saved CSVs."""
    renames = get_state_renames(dataset_name)
    return {
        node: [renames.get(state, state) for state in states]
        for node, states in model.states.items()
    }


def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError("The model must be an instance of LinearGaussianBayesianNetwork")

    samples = pd.DataFrame(columns=list(model.nodes()))
    for node in nx.topological_sort(model):
        cpd = model.get_cpds(node)

        if cpd.evidence:
            values = np.vstack([samples[parent] for parent in cpd.evidence])
            mean = cpd.beta[0] + np.dot(cpd.beta[1:], values)
            samples[node] = rng.normal(mean, cpd.std)
        else:
            samples[node] = rng.normal(cpd.beta[0], cpd.std, size=(num_samples,))

    return samples


def sample_from_discrete(model, num_samples, rng=default_rng(), **kwargs):
    """Sample from a discrete model using ancestral sampling."""
    if not isinstance(model, BayesianNetwork):
        raise ValueError("The model must be an instance of BayesianNetwork")
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples, show_progress=False, **kwargs)

    # Convert values to pd.Categorical for faster operations
    for node in samples.columns:
        cpd = model.get_cpds(node)
        samples[node] = pd.Categorical(samples[node], categories=cpd.state_names[node])

    return samples
