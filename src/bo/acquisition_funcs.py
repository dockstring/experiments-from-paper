import numpy as np
from scipy import stats


def upper_confidence_bound(mu: np.array, var: np.array, beta: float):
    return mu + np.sqrt(beta * var)


def expected_improvement(mu: np.array, var: np.array, y_best: float):
    std = np.sqrt(var)
    z = (mu - y_best) / std
    return (mu - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
