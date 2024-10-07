import numpy as np
from qutlet.utilities import gaussian_envelope
import matplotlib.pyplot as plt


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


def get_exp_filter_function(a) -> callable:
    def filter_function(time: float):
        return np.exp(-a * time)

    return filter_function


def get_fourier_gaps_filter_function(
    sys_eig_energies: np.ndarray, width: float = 0.001, freq_positivity: str = "pos"
) -> callable:
    n = int(1e3)
    spectrum_width = np.abs(np.max(sys_eig_energies) - np.min(sys_eig_energies))
    mus = np.array(
        [
            np.abs(sys_eig_energies[ind] - sys_eig_energies[0])
            for ind in range(1, len(sys_eig_energies))
        ]
    )
    _, indices = np.unique(np.round(mus, 8), return_index=True)

    mus = mus[indices]
    peaks = np.sum(
        [
            gaussian_envelope(mu=mu / spectrum_width, sigma=width, n_steps=n)
            for mu in mus
        ],
        axis=0,
    )

    if freq_positivity == "neg":
        peaks = np.concatenate((np.zeros_like(peaks), peaks[::-1]))
    elif freq_positivity == "pos":
        peaks = np.concatenate((peaks, np.zeros_like(peaks)))
    elif freq_positivity == "both":
        peaks = np.concatenate((peaks, peaks[::-1]))
    sampling_rate = 2 * np.max(mus)
    n = 3 * sampling_rate
    filter_array = np.fft.ifft(peaks)

    times = np.arange(0, n / sampling_rate, 1 / sampling_rate)

    def filter_function(time: float):
        # Whittaker–Shannon
        return np.abs(
            np.sum(
                np.array(
                    [
                        filter_array[ind] * np.sinc((time - times[ind]) / times[ind])
                        for ind in range(1, len(times))
                    ]
                )
            )
        )

    return filter_function
