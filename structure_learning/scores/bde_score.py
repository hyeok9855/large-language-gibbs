"""
Adapted from
https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/scores/bde_score.py
"""

import math

import numpy as np
import pandas as pd
from scipy.special import gammaln

from structure_learning.priors.base import BasePrior
from .base import BaseScore, LocalScore


class BDeScore(BaseScore):
    """BDe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (discrete) dataset D. Each column
        corresponds to one variable. If there is interventional data, the
        interventional targets must be specified in the "INT" column (the
        indices of interventional targets are assumed to be 1-based).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    equivalent_sample_size : float (default: 1.)
        The equivalent sample size (of uniform pseudo samples) for the
        Dirichlet hyperparameters. The score is sensitive to this value,
        runs with different values might be useful.
    """

    def __init__(self, data: pd.DataFrame, prior: BasePrior, equivalent_sample_size: float = 1.0):
        if "INT" in data.columns:  # Interventional data
            # Indices should start at 0, instead of 1;
            # observational data will have INT == -1.
            self._interventions = data.INT.map(lambda x: int(x) - 1)
            data = data.drop(["INT"], axis=1)
        else:
            self._interventions = np.full(data.shape[0], -1)
        super().__init__(data, prior)
        self.equivalent_sample_size = equivalent_sample_size

        assert self.data is not None
        self.state_names = {
            column: sorted(self.data[column].cat.categories.tolist())
            for column in self.data.columns
        }

    def _local_score(self, target: int, indices: tuple[int, ...]) -> LocalScore:
        counts = self._state_counts(target, indices)
        num_parents_states = counts.shape[1]

        log_gamma_counts = np.zeros_like(counts, dtype=np.float64)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size

        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float64)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        local_score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * math.lgamma(alpha)
            - counts.size * math.lgamma(beta)
        )

        return LocalScore(
            key=(target, tuple(indices)),
            score=local_score,
            prior=self.prior.local_score(target, indices),
        )

    def _state_counts(self, target: int, indices: tuple[int, ...]) -> np.ndarray:
        # Source: pgmpy.estimators.BaseEstimator.state_counts()
        # TODO: use numpy instead of pandas

        parents = [self.column_names[index] for index in indices]
        variable = self.column_names[target]

        assert self.data is not None
        data = self.data[self._interventions != target]
        data = data[[variable] + parents].dropna()

        state_count_data = data.groupby([variable] + parents).size().unstack(parents)

        if len(parents) == 0:
            return state_count_data.fillna(0).values[:, None]

        if not isinstance(state_count_data.columns, pd.MultiIndex):
            state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])

        parent_states = [self.state_names[parent] for parent in parents]
        columns_index = pd.MultiIndex.from_product(parent_states, names=parents)

        return (
            state_count_data.reindex(index=self.state_names[variable], columns=columns_index)
            .fillna(0)
            .values
        )
