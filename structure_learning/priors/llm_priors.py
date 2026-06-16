"""
Adapted from https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/scores/bge_score.py
and also https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/scores/bde_score.py
"""

from abc import ABC

import pandas as pd
import numpy as np

from .base import BasePrior
from .uninformative_priors import UninformativePrior
from structure_learning.scores.bde_score import BDeScore
from structure_learning.scores.bge_score import BGeScore


class LLMDataPrior(BasePrior, ABC):
    """
    Prior with LLM-generated data as input.

    Args:
        data: LLM-generated data.
        uninformative_prior: Uninformative prior (e.g., UniformPrior).
        gamma: inverse temperature in [0, 1] for the prior. Default is 1.0.
            0 reduces to the uninformative prior, 1.0 corresponds to standard Bayesian update.
    """

    def __init__(
        self,
        num_variables: int,
        data: pd.DataFrame,
        uninformative_prior: UninformativePrior,
        gamma: float = 1.0,
    ):
        super().__init__(num_variables)
        assert num_variables == len(data.columns) == uninformative_prior.num_variables
        self.data = data
        self.uninformative_prior = uninformative_prior
        self.gamma = gamma


class LLMDataBGePrior(LLMDataPrior):
    """
    Prior with LLM-generated data as input and BGe score.

    Args:
        data: LLM-generated data.
        uninformative_prior: Uninformative prior (e.g., UniformPrior).
        gamma: inverse temperature in [0, 1] for the prior. Default is 1.0.
            0 reduces to the uninformative prior, 1.0 corresponds to standard Bayesian update.
        mean_obs: np.ndarray of size (num_variables,). Mean of the observed data.
            If None, we set it to 0 for all variables.
        alpha_mu: float. Parameter $\alpha_{\mu}$ corresponding to the precision parameter
            of the Normal prior over the mean $\mu$.
        alpha_w: float. Parameter $\alpha_{w}$ corresponding to the number of degrees of
            freedom of the Wishart prior of the precision matrix $W$. This parameter must satisfy
            `alpha_w > N - 1`, where `N` is the number of varaibles. By default, `alpha_w = N + 2`.
    """

    def __init__(
        self,
        num_variables: int,
        data: pd.DataFrame,
        uninformative_prior: UninformativePrior,
        gamma: float = 1.0,
        mean_obs: np.ndarray | None = None,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
    ):
        super().__init__(num_variables, data, uninformative_prior, gamma)

        self.bge_scorer = BGeScore(
            data=self.data,
            prior=self.uninformative_prior,
            mean_obs=mean_obs,
            alpha_mu=alpha_mu,
            alpha_w=alpha_w,
        )

    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        local_score = self.bge_scorer.local_score(target, indices)
        return self.gamma * local_score.score + local_score.prior


class LLMDataBDePrior(LLMDataPrior):
    """
    Prior with LLM-generated data as input and BDe score.

    Args:
        data: LLM-generated data.
        uninformative_prior: Uninformative prior (e.g., UniformPrior).
        gamma: inverse temperature in [0, 1] for the prior. Default is 1.0.
            0 reduces to the uninformative prior, 1.0 corresponds to standard Bayesian update.
        equivalent_sample_size: float. The equivalent sample size (of uniform pseudo samples) for
            the Dirichlet hyperparameters. The score is sensitive to this value, runs with
            different values might be useful.
    """

    def __init__(
        self,
        num_variables: int,
        data: pd.DataFrame,
        uninformative_prior: UninformativePrior,
        gamma: float = 1.0,
        equivalent_sample_size: float = 1.0,
    ):
        super().__init__(num_variables, data, uninformative_prior, gamma)

        self.bde_scorer = BDeScore(
            data=self.data,
            prior=self.uninformative_prior,
            equivalent_sample_size=equivalent_sample_size,
        )

    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        local_score = self.bde_scorer.local_score(target, indices)
        return self.gamma * local_score.score + local_score.prior


class LLMEdgeMatrixPrior(BasePrior, ABC):
    """
    Prior with LLM-generated edge matrix as input.

    Args:
        edge_matrix: LLM-generated edge matrix.
    """

    def __init__(self, num_variables: int, edge_matrix: np.ndarray):
        assert edge_matrix.shape == (num_variables, num_variables)
        super().__init__(num_variables)
        self.edge_matrix = edge_matrix


class LLMEdgeMatrixBernoulliPrior(LLMEdgeMatrixPrior):
    """
    Prior with LLM-generated edge matrix as input.

    Args:
        edge_matrix: LLM-generated edge matrix.
        uniform_mix_ratio: float. The ratio of the uniform prior to the LLM-generated prior.
    """

    def __init__(
        self,
        num_variables: int,
        edge_matrix: np.ndarray,
        uniform_mix_ratio: float = 0.5,
    ) -> None:
        super().__init__(num_variables, edge_matrix)
        self.uniform_mix_ratio = uniform_mix_ratio
        uniform_matrix = np.ones((self.num_variables, self.num_variables)) * 0.5
        uniform_matrix[np.diag_indices(self.num_variables)] = 0.0
        self.M = (1 - uniform_mix_ratio) * self.edge_matrix + uniform_mix_ratio * uniform_matrix

    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        # Assuming independent Bernoulli edge probabilities, the score (excluding the constant) is:
        # S_i(Pa_i) = \sum_{j \in Pa_i} (\log(M_{j,i}) - \log(1 - M_{j,i}))
        return (np.log(self.M[indices, target]) - np.log(1 - self.M[indices, target])).sum().item()


class LLMEdgeMatrixL1Prior(LLMEdgeMatrixPrior):
    """
    Prior with LLM-generated edge matrix as input and L1 prior.

    Args:
        edge_matrix: LLM-generated edge matrix.
        l1_lambda: float. The lambda parameter for the L1 prior.
    """

    def __init__(self, num_variables: int, edge_matrix: np.ndarray, l1_lambda: float = 1.0):
        super().__init__(num_variables, edge_matrix)
        self.l1_lambda = l1_lambda
        self.one_minus_two_M = 1.0 - 2.0 * self.edge_matrix

    def local_score(self, target: int, indices: tuple[int, ...]) -> float:
        # S0(G) = -\lambda \sum_{i \neq j} |M_{i,j} - G_{i,j}|
        #       = -\lambda \sum_{(i,j) \in G} (1 - 2 * M_{i,j}) + const.
        # S((i,j)) = -lambda * (1 - 2 * M_{i,j})
        return -self.l1_lambda * self.one_minus_two_M[indices, target].sum().item()
