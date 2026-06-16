import numpy as np


def logdet(array: np.ndarray) -> float:
    _, logdet = np.linalg.slogdet(array)
    return logdet
