"""
Adapted from:
https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/scores/priors.py
"""

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gammaln

from .base import BasePrior


class UninformativePrior(BasePrior, ABC):
    def __init__(self, num_variables: int):
        super().__init__(num_variables)
        self._log_prior = None

    @property
    @abstractmethod
    def log_prior(self) -> np.ndarray:
        pass

    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        return self.log_prior[len(indices)].item()


class UniformPrior(UninformativePrior):
    @property
    def log_prior(self) -> np.ndarray:
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_variables,))
        return self._log_prior


class ErdosRenyiPrior(UninformativePrior):
    def __init__(self, num_variables: int, num_edges_per_node: float = 1.0):
        super().__init__(num_variables)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self) -> np.ndarray:
        if self._log_prior is None:
            num_edges = self.num_variables * self.num_edges_per_node  # Default value
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            all_parents = np.arange(self.num_variables)
            self._log_prior = all_parents * math.log(p) + (
                self.num_variables - all_parents - 1
            ) * math.log1p(-p)
        return self._log_prior


class EdgePrior(UninformativePrior):
    def __init__(self, num_variables: int, beta: float = 1.0):
        super().__init__(num_variables)
        self.beta = beta

    @property
    def log_prior(self) -> np.ndarray:
        if self._log_prior is None:
            self._log_prior = np.arange(self.num_variables) * math.log(self.beta)
        return self._log_prior


class FairPrior(UninformativePrior):
    @property
    def log_prior(self) -> np.ndarray:
        if self._log_prior is None:
            all_parents = np.arange(self.num_variables)
            self._log_prior = (
                -gammaln(self.num_variables + 1)
                + gammaln(self.num_variables - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        return self._log_prior
