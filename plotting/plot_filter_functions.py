from fermionic_cooling.filter_functions import (
    get_ding_filter_function,
    get_lloyd_filter_function,
)

import matplotlib.pyplot as plt
import numpy as np


def evaluate_and_fft(model: callable, times: np.ndarray):
    y = np.array([model(t) for t in times])
    y[np.isnan(y)] = np.nanmax(y)
    y /= np.max(y)
    fy = np.abs(np.fft.fft(y))
    fy /= np.max(fy)
    return y, fy


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

    fig, axes = plt.subplots(nrows=2)

    lloyd_y, lloyd_fy = evaluate_and_fft(lloyd, times)
    ding_y, ding_fy = evaluate_and_fft(ding, times)

    axes[0].plot(times, lloyd_y, label="lloyd")
    axes[0].plot(times, ding_y, label="ding")
    axes[0].legend()
    axes[0].set_xlabel("t")

    axes[1].plot(freq, lloyd_fy, label="lloyd")
    axes[1].plot(freq, ding_fy, label="ding")
    axes[1].legend()
    axes[1].set_xlabel("$\omega$")

    plt.show()


if __name__ == "__main__":
    main()
