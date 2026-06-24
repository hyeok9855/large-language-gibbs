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

        # Build robust code mapping that guarantees alignment with sorted categories
        data_codes_list = []
        self.cardinalities = []
        for column in self.data.columns:
            cats = self.data[column].cat.categories.tolist()
            sorted_cats = sorted(cats)
            self.cardinalities.append(len(sorted_cats))

            # map from original pandas category code -> sorted category index
            mapping = np.array([sorted_cats.index(cat) for cat in cats], dtype=np.int64)

            orig_codes = self.data[column].cat.codes.values
            mapped_codes = np.where(orig_codes >= 0, mapping[orig_codes], -1)
            data_codes_list.append(mapped_codes)

        self.data_codes = np.array(data_codes_list, dtype=np.int64).T
        self.cardinalities = np.array(self.cardinalities, dtype=np.int64)

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
        # Filter interventional data for target node
        mask = self._interventions != target

        cols = [target] + list(indices)
        filtered_codes = self.data_codes[mask][:, cols]

        # Drop rows with NaNs (codes == -1) to match original dropna() behavior
        valid_mask = np.all(filtered_codes >= 0, axis=1)
        filtered_codes = filtered_codes[valid_mask]

        card_t = self.cardinalities[target]
        target_codes = filtered_codes[:, 0]
        if len(indices) == 0:
            counts = np.bincount(target_codes, minlength=card_t)
            return counts[:, None]
        parent_codes = filtered_codes[:, 1:]
        parent_cards = self.cardinalities[list(indices)]

        # Strides for row-major product indexing
        strides = np.r_[parent_cards[1:], 1]
        strides = np.cumprod(strides[::-1])[::-1]

        parent_index = np.dot(parent_codes, strides)
        num_parent_states = np.prod(parent_cards)

        # Map 2D coordinate (target, parent_index) to 1D index
        flat_index = target_codes * num_parent_states + parent_index
        flat_counts = np.bincount(flat_index, minlength=card_t * num_parent_states)
        return flat_counts.reshape(card_t, num_parent_states)
