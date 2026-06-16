"""
The code is adapted from:
https://github.com/larslorch/dibs/blob/master/dibs/metrics.py and
https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/metrics.py
"""

import numpy as np

from sklearn import metrics


def expected_shd(posterior, ground_truth):
    """Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    e_shd : float
        The Expected SHD.
    """
    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def expected_edges(posterior):
    """Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    Returns
    -------
    e_edges : float
        The expected number of edges.
    """
    num_edges = np.sum(posterior, axis=(1, 2))
    return np.mean(num_edges)


def threshold_metrics(posterior, ground_truth):
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    metrics : dict
        The threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(posterior, axis=0)
    p_edge_flat = p_edge.reshape(-1)

    gt_flat = ground_truth.reshape(-1)

    # Threshold metrics
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "prc_auc": prc_auc,
        "ave_prec": ave_prec,
    }
