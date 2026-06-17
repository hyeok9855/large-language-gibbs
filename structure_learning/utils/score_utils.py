"""
Adapted from:
https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/data.py
"""

import gzip
import string
import urllib.request
from itertools import chain, product, islice, count
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.example_models import load_model
from pgmpy.utils import get_example_model

from structure_learning.priors import (
    BasePrior,
    UniformPrior,
    ErdosRenyiPrior,
    EdgePrior,
    FairPrior,
    LLMDataBGePrior,
    LLMDataBDePrior,
    LLMEdgeMatrixBernoulliPrior,
    LLMEdgeMatrixL1Prior,
)
from structure_learning.scores import BDeScore, BGeScore
from structure_learning.utils.bn_utils import sample_from_linear_gaussian


FILE_DIR = Path(__file__).parent
DATA_DIR = FILE_DIR.parent / "datasets"


def sample_erdos_renyi_graph(
    num_variables,
    p=None,
    num_edges=None,
    nodes=None,
    create_using=models.BayesianNetwork,
    rng=default_rng(),
):
    if p is None:
        if num_edges is None:
            raise ValueError("One of p or num_edges must be specified.")
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.0)

    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(product(uppercase, repeat=r) for r in count(1))
        nodes = ["".join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)  # type: ignore
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_erdos_renyi_linear_gaussian(
    num_variables,
    p=None,
    num_edges=None,
    nodes=None,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_noise=0.1,
    rng=default_rng(),
):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,  # type: ignore
        rng=rng,
    )

    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.0  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, "wb") as f:
        f.write(file_content)

    return filename


def get_data(name, args, rng=default_rng()):
    if name == "erdos_renyi_lingauss":
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng,
        )
        data = sample_from_linear_gaussian(graph, num_samples=args.num_samples, rng=rng)
        score = "bge"

    elif name == "sachs_continuous":
        graph = get_example_model("sachs")
        assert graph is not None
        filename = download(
            "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz", Path("data/sachs.data.txt")
        )
        data = pd.read_csv(filename, delimiter="\t", dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = "bge"

    elif name == "sachs_interventional":
        graph = get_example_model("sachs")
        assert graph is not None
        filename = download(
            "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz",
            Path("data/sachs.interventional.txt"),
        )
        data = pd.read_csv(filename, delimiter=" ", dtype="category")
        score = "bde"

    elif name == "bn":
        graph = load_model(args.dataset_name.replace("_", "/"))
        if args.data_path is not None:
            data = pd.read_csv(args.data_path, dtype="category")
        else:
            assert args.num_samples is not None and args.data_seed is not None
            data = pd.read_csv(
                DATA_DIR / args.dataset_name / f"data_n{args.num_samples}_sd{args.data_seed}.csv",
                dtype="category",
            )
        # reorder the columns to match the graph
        data = data[list(graph.nodes())]
        score = "bde"
    else:
        raise ValueError(f"Unknown graph type: {args.dataset_name}")

    return graph, data, score


def get_prior(
    name: str, score: str, num_variables: int, nodes: list[str] | None = None, **kwargs
) -> BasePrior:
    prior = {
        "uniform": UniformPrior,
        "erdos_renyi": ErdosRenyiPrior,
        "edge": EdgePrior,
        "fair": FairPrior,
        "llm_data_bge": LLMDataBGePrior,
        "llm_data_bde": LLMDataBDePrior,
        "llm_edge_matrix_bernoulli": LLMEdgeMatrixBernoulliPrior,
        "llm_edge_matrix_l1": LLMEdgeMatrixL1Prior,
    }
    if name == "llm_data":
        name = f"llm_data_{score}"
        kwargs["data"] = pd.read_csv(
            kwargs["data_path"], dtype="category" if score == "bde" else float
        )
        kwargs["base_prior"] = get_prior(
            name=kwargs["base_prior"],
            score=score,
            num_variables=num_variables,
            **kwargs.get("base_prior_kwargs", {}),
        )
        del kwargs["data_path"]
        if "base_prior_kwargs" in kwargs:
            del kwargs["base_prior_kwargs"]

    if name.startswith("llm_edge_matrix_"):
        if "edge_matrix_path" in kwargs:
            path = Path(kwargs["edge_matrix_path"])
            assert path.suffix == ".csv"
            df = pd.read_csv(path, index_col=0)
            if nodes is not None:
                # Ensures the rows and columns perfectly match the expected graph / dataset ordering
                df = df.reindex(index=nodes, columns=nodes)
                if df.isnull().values.any():
                    raise ValueError("Some nodes in the graph are not found in the edge matrix.")
            kwargs["edge_matrix"] = df.values
            del kwargs["edge_matrix_path"]

    return prior[name](num_variables=num_variables, **kwargs)


def get_scorer(args, rng=default_rng()):
    # Get the data
    graph, data, score = get_data(args.graph, args, rng=rng)

    # Get the prior
    prior = get_prior(
        args.prior,
        score,
        num_variables=graph.order(),
        nodes=list(graph.nodes()),
        **args.prior_kwargs,
    )

    # Get the scorer
    scores = {"bde": BDeScore, "bge": BGeScore}
    scorer = scores[score](data, prior, **args.scorer_kwargs)

    return scorer, data, graph
