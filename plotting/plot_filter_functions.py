from fermionic_cooling.filter_functions import (
    get_ding_filter_function,
    get_lloyd_filter_function,
    get_fourier_gaps_filter_function,
)

import matplotlib.pyplot as plt
import numpy as np
from qutlet.models import FermiHubbardModel


def evaluate_and_fft(model: callable, times: np.ndarray):
    y = np.array([model(t) for t in times])
    y[np.isnan(y)] = np.nanmax(y)
    y /= np.max(y)
    fy = np.abs(np.fft.fft(y))
    fy /= np.max(fy)
    return y, fy


def get_fh_ff(freq_positivity):
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), tunneling=1, coulomb=6, n_electrons="hf"
    )
    energies, states = model.spectrum
    return get_fourier_gaps_filter_function(energies, freq_positivity=freq_positivity)


def main():
    spectrum_width = 15.269206109814778
    min_gap = 1.9999999999999964

    time = 2.58
    times = np.linspace(0.0001, time, 1001)
    freq = np.fft.fftfreq(times.shape[-1])

    lloyd = get_lloyd_filter_function(biga=1, beta=time / 3, tau=time / 2)
    ding = get_ding_filter_function(
        a=2.5 * spectrum_width, da=0.5 * spectrum_width, b=min_gap, db=min_gap
    )
    gaps_pos = get_fh_ff("pos")
    gaps_neg = get_fh_ff("neg")
    gaps_both = get_fh_ff("both")

    ffunctions = (
        [lloyd, "lloyd"],
        [ding, "ding"],
        [gaps_pos, "g pos"],
        [gaps_neg, "g neg"],
        [gaps_both, "g both"],
    )
    fig, axes = plt.subplots(nrows=2)

    for ind, (ff, lab) in enumerate(ffunctions):

        y, fy = evaluate_and_fft(ff, times)

        axes[0].plot(times, y, label=lab)
        axes[0].legend()
        axes[0].set_xlabel("t")

        axes[1].plot(freq, fy, label=lab)
        axes[1].legend()
        axes[1].set_xlabel("$\omega$")

    plt.show()


if __name__ == "__main__":
    main()
