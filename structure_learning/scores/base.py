import sys
from abc import ABC, abstractmethod
from collections import namedtuple

import networkx as nx
import pandas as pd
from pgmpy.estimators import StructureScore
from functools import lru_cache

from structure_learning.priors.base import BasePrior


LocalScore = namedtuple("LocalScore", ["key", "score", "prior"])


class BaseScore(StructureScore, ABC):
    """Base class for the scorer.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """

    def __init__(self, data: pd.DataFrame, prior: BasePrior):
        StructureScore.__init__(self, data)
        self.prior = prior
        self.num_samples, self.num_variables = data.shape
        assert self.num_variables == self.prior.num_variables

        self.column_names = list(data.columns)
        self.column_names_to_idx = dict((name, idx) for (idx, name) in enumerate(self.column_names))

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data

                local_score: LocalScore = self.local_score(target, indices)
                out_queue.put((True, *local_score))

                if indices_after is not None:
                    local_score_after: LocalScore = self.local_score(target, indices_after)
                    out_queue.put((True, *local_score_after))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None, None))

    @lru_cache(maxsize=100_000)
    def local_score(self, target: int, indices: tuple[int, ...]) -> LocalScore:
        return self._local_score(target, indices)

    @abstractmethod
    def _local_score(self, target: int, indices: tuple[int, ...]) -> LocalScore:
        pass

    def score(self, model: nx.DiGraph) -> float:
        # graph = nx.relabel_nodes(model, self.column_names_to_idx)
        score = 0.0
        for node in model.nodes():
            node_idx = self.column_names_to_idx[node]
            predecessors_idx = tuple(
                self.column_names_to_idx[pred] for pred in sorted(model.predecessors(node))
            )
            local_score = self.local_score(node_idx, predecessors_idx)
            score += local_score.score + local_score.prior
        return score
