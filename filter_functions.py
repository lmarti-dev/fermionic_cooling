import numpy as np


def get_ding_filter_function(
    a: float, da: float, b: float, db: float, take_abs: bool = True
):
    # Single-ancilla ground state preparation via Lindbladians
    # paper values
    # a=2.5 * spectrum_width, da=0.5 * spectrum_width, b=min_gap (spectral gap), db=min_gap
    if take_abs:

        def filter_function(time: float):
            pa = np.exp((-(time**2) * (da**2) / 4)) * np.exp(1j * a * time)
            pb = np.exp((-(time**2) * (db**2) / 4)) * np.exp(1j * b * time)
            return np.abs((pa - pb) / (2 * np.pi * 1j * time))

    else:

        def filter_function(time: float):
            pa = np.exp((-(time**2) * (da**2) / 4)) * np.exp(1j * a * time)
            pb = np.exp((-(time**2) * (db**2) / 4)) * np.exp(1j * b * time)
            return (pa - pb) / (2 * np.pi * 1j * time)

    return filter_function


def get_lloyd_filter_function(biga: float, beta: float, tau: float):
    # Quasiparticle cooling algorithms for quantum many-body state preparation
    # arXiv:2404.12175v1 [quant-ph] 18 Apr 2024
    def filter_function(time: float):
        return np.sin((np.pi * (time - tau) / 2)) / (
            (beta) * (np.sinh(np.pi * (time - tau) / beta))
        )

    return filter_function
