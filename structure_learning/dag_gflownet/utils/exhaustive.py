import numpy as np
import networkx as nx

from scipy.special import logsumexp
from dataclasses import dataclass
from collections import defaultdict
from typing import DefaultDict
from pgmpy.utils.mathext import powerset
from itertools import permutations, product
from tqdm import tqdm

# from structure_learning.utils.nx_utils import get_markov_blanket_graph

# https://oeis.org/A003024
NUM_DAGS = [1, 1, 3, 25, 543, 29281, 3781503]


class GraphCollection:
    def __init__(self):
        # Mutable (building) state
        self._edges_list: list[int] = []
        self._lengths_list: list[int] = []
        self._mapping_dd: DefaultDict[tuple[int, int], int] = defaultdict(int)
        self._mapping_dd.default_factory = lambda: len(self._mapping_dd)

        # Frozen state (after calling freeze)
        self.edges: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.mapping: list[tuple[int, int]] | None = None

    def append(self, graph):
        if self.is_frozen():
            raise RuntimeError("GraphCollection is frozen; cannot append.")
        self._edges_list.extend([self._mapping_dd[edge] for edge in graph.edges()])
        self._lengths_list.append(graph.number_of_edges())

    def freeze(self):
        if self.is_frozen():
            return self
        self.edges = np.asarray(self._edges_list, dtype=np.int_)
        self.lengths = np.asarray(self._lengths_list, dtype=np.int_)
        self.mapping = [edge for (edge, _) in sorted(self._mapping_dd.items(), key=lambda x: x[1])]
        return self

    def is_frozen(self):
        return self.mapping is not None

    def to_dict(self, prefix=None):
        if not self.is_frozen():
            raise ValueError('Graphs must be frozen. Call "graphs.freeze()".')
        prefix = f"{prefix}_" if (prefix is not None) else ""
        return {
            f"{prefix}edges": self.edges,
            f"{prefix}lengths": self.lengths,
            f"{prefix}mapping": self.mapping,
        }

    def load(self, edges, lengths, mapping):
        # Load directly into frozen representation
        self.edges = edges
        self.lengths = lengths
        self.mapping = list(mapping)
        return self


@dataclass
class FullPosterior:
    log_probas: np.ndarray
    graphs: GraphCollection
    # closures: GraphCollection
    # markov: GraphCollection

    def to_dict(self):
        # Ensure that "graphs" has been frozen
        if not self.graphs.is_frozen():
            raise ValueError('Graphs must be frozen. Call "graphs.freeze()".')

        assert (
            self.graphs.lengths is not None
            and self.graphs.edges is not None
            and self.graphs.mapping is not None
        )

        offset, output = 0, dict()
        for length, log_prob in zip(self.graphs.lengths, self.log_probas):
            edges_indices = self.graphs.edges[offset : offset + length]
            edges = [self.graphs.mapping[idx] for idx in edges_indices]
            output[frozenset(edges)] = log_prob
            offset += length

        return output

    def save(self, filename):
        with open(filename, "wb") as f:
            np.savez(
                f,
                log_probas=self.log_probas,
                **self.graphs.to_dict(prefix="graphs"),
                # **self.closures.to_dict(prefix="closures"),
                # **self.markov.to_dict(prefix="markov"),
            )

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            data = np.load(f)
            log_probas = data["log_probas"]
            graphs = GraphCollection().load(
                data["graphs_edges"], data["graphs_lengths"], data["graphs_mapping"]
            )
            # closures = GraphCollection().load(
            #     data["closures_edges"], data["closures_lengths"], data["closures_mapping"]
            # )
            # markov = GraphCollection().load(
            #     data["markov_edges"], data["markov_lengths"], data["markov_mapping"]
            # )
        return cls(
            log_probas=log_probas,
            graphs=graphs,
            # closures=closures,
            # markov=markov,
        )


def get_full_posterior(data, scorer, verbose=True):
    log_probas = []
    graphs = GraphCollection()
    # closures = GraphCollection()
    # markov = GraphCollection()

    for graph in tqdm(
        all_dags(data.shape[1], nodelist=sorted(data.columns)),
        desc="[get_full_posterior]",
        total=NUM_DAGS[data.shape[1]],
        disable=(not verbose),
        dynamic_ncols=True,
    ):
        score = scorer.score(graph)
        log_probas.append(score)
        graphs.append(graph)
        # closures.append(nx.transitive_closure_dag(graph))
        # markov.append(get_markov_blanket_graph(graph))

    # Normalize the log-joint distribution to get the posterior
    log_probas = np.asarray(log_probas, dtype=np.float64)
    log_probas -= logsumexp(log_probas)

    return FullPosterior(
        log_probas=log_probas,
        graphs=graphs.freeze(),
        # closures=closures.freeze(),
        # markov=markov.freeze(),
    )


def get_gfn_exact_posterior(gfn_state_graph, verbose=True):
    # Get the source graph
    in_degrees = gfn_state_graph.in_degree(gfn_state_graph)
    source_graphs = [
        gfn_state_graph.nodes[node]["graph"] for node, in_degree in in_degrees if in_degree == 0
    ]
    assert len(source_graphs) == 1
    assert len(source_graphs[0].edges) == 0
    num_variables = len(source_graphs[0])

    log_probas = []
    graphs = GraphCollection()
    # closures = GraphCollection()
    # markov = GraphCollection()

    for node in tqdm(
        nx.topological_sort(gfn_state_graph),
        desc="[get_gfn_exact_posterior]",
        total=NUM_DAGS[num_variables],
        disable=(not verbose),
        dynamic_ncols=True,
    ):
        graph = gfn_state_graph.nodes[node]["graph"]
        log_probas.append(gfn_state_graph.nodes[node]["terminal_log_flow"])
        graphs.append(graph)
        # closures.append(nx.transitive_closure_dag(graph))
        # markov.append(get_markov_blanket_graph(graph))

    # The log-posterior is already normalized
    log_probas = np.asarray(log_probas, dtype=np.float64)

    return FullPosterior(
        log_probas,
        graphs=graphs.freeze(),
        # closures=closures.freeze(),
        # markov=markov.freeze(),
    )


def _get_log_features(graphs, log_probas):
    indices = np.zeros_like(graphs.lengths)
    indices[1:] = np.cumsum(graphs.lengths[:-1])

    features = dict()
    for index, edge in enumerate(graphs.mapping):
        if not np.any(graphs.edges == index):
            continue
        has_feat = np.add.reduceat(graphs.edges == index, indices)

        # Edge case: the first graph is the empty graph, it has no edge
        if graphs.lengths[0] == 0:
            has_feat[0] = 0
        assert np.sum(graphs.edges == index) == np.sum(has_feat)

        has_feat = has_feat.astype(np.bool_)
        features[edge] = logsumexp(log_probas[has_feat])

    return features


def get_edge_log_features(posterior):
    return _get_log_features(posterior.graphs, posterior.log_probas)


# def get_path_log_features(posterior):
#     return _get_log_features(posterior.closures, posterior.log_probas)


# def get_markov_blanket_log_features(posterior):
#     return _get_log_features(posterior.markov, posterior.log_probas)


def all_dags(num_variables, nodelist=None):
    if nodelist is None:
        nodelist = list(range(num_variables))

    G = nx.DiGraph()
    G.add_nodes_from(nodelist)

    # Must match the edge generation order of permutations
    possible_edges = list(permutations(nodelist, 2))
    max_edges = len(possible_edges)

    def backtrack_by_size(idx, k):
        # Base case: target edge count reached
        if k == 0:
            yield G.copy()
            return

        # Pruning: not enough edges left to reach target k
        if idx == max_edges or max_edges - idx < k:
            return

        u, v = possible_edges[idx]

        # Branch 1: INCLUDE (This must come first to match itertools.combinations order)
        if not nx.has_path(G, v, u):
            G.add_edge(u, v)
            yield from backtrack_by_size(idx + 1, k - 1)
            G.remove_edge(u, v)

        # Branch 2: EXCLUDE
        yield from backtrack_by_size(idx + 1, k)

    # Mimic powerset by iterating through all possible sizes lengths
    for r in range(max_edges + 1):
        yield from backtrack_by_size(0, r)


def all_hashes(num_variables):
    hashes = {edge: 2**i for (i, edge) in enumerate(product(range(num_variables), repeat=2))}
    for graph in all_dags(num_variables):
        yield sum(hashes[edge] for edge in graph.edges)


###############
# Legacy code #
###############


def all_dags_legacy(num_variables, nodelist=None):
    # Adapted from: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/ExhaustiveSearch.py
    if nodelist is None:
        nodelist = list(range(num_variables))
    edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    all_graphs = powerset(edges)  # 2^(n*(n-1)) graphs

    for graph_edges in all_graphs:
        graph = nx.DiGraph(graph_edges)
        graph.add_nodes_from(nodelist)
        if nx.is_directed_acyclic_graph(graph):
            yield graph


if __name__ == "__main__":

    import time

    num_variables = 5
    start_time = time.time()
    all_dags_backtracking = list(all_dags(num_variables))
    end_time = time.time()
    print(f"Time taken to generate all the DAGs (backtracking): {end_time - start_time} seconds")
    start_time = time.time()
    _all_dags_legacy = list(all_dags_legacy(num_variables))
    end_time = time.time()
    print(f"Time taken to generate all the DAGs (legacy): {end_time - start_time} seconds")

    # Check if they are the same length
    assert len(_all_dags_legacy) == len(all_dags_backtracking)

    # Check if they are the same
    for dag_legacy, dag_new in zip(_all_dags_legacy, all_dags_backtracking):
        assert frozenset(dag_legacy.edges()) == frozenset(dag_new.edges())
