import numpy as np


def gaussian_bell(mu: float, sigma: float, n_steps: int):
    return np.exp(-((np.linspace(-1, 1, n_steps) - mu) ** 2) / (2 * sigma**2))


b = gaussian_bell(1, 0.1, 35)


print(b)
