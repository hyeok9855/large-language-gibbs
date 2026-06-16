import numpy as np


def jensen_shannon_divergence(full_posterior, posterior):
    # Convert to dictionaries to align distributions
    full_posterior_dict = full_posterior.to_dict()
    posterior_dict = posterior.to_dict()

    # Get an (arbitrary ordering of the graphs)
    graphs = list(full_posterior_dict.keys())
    graphs = sorted(graphs, key=len)

    # Get the two distributions aligned
    full_posterior, posterior = [], []
    for graph in graphs:
        full_posterior.append(full_posterior_dict[graph])
        posterior.append(posterior_dict[graph])
    full_posterior = np.array(full_posterior, dtype=np.float64)
    posterior = np.array(posterior, dtype=np.float64)

    # Compute the mean distribution
    mean = np.log(0.5) + np.logaddexp(full_posterior, posterior)

    # Compute the JSD
    KL_full_posterior = np.exp(full_posterior) * (full_posterior - mean)
    KL_posterior = np.exp(posterior) * (posterior - mean)
    return 0.5 * np.sum(KL_full_posterior + KL_posterior)
