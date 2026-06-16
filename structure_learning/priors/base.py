"""
Adapted from:
https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/scores/base.py
"""

from abc import ABC, abstractmethod

import networkx as nx


class BasePrior(ABC):
    """Base class for the prior over graphs p(G).

    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.

    Args:
        num_variables: int
            The number of variables in the graph.
    """

    def __init__(self, num_variables: int):
        self.num_variables = num_variables

    @abstractmethod
    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        """
        Return a local prior score for a given node (target) and its parents (indices).
        """
        pass

    def score(self, graph: nx.DiGraph) -> float:
        """
        Return a prior score P(G) for a given graph G.
        """
        assert set(graph.nodes()) == set(range(self.num_variables))

        score = 0
        for node in graph.nodes():
            local_score = self.local_score(node, tuple(graph.predecessors(node)))
            score += local_score
        return score
