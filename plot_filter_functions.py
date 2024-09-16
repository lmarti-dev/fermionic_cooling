from fermionic_cooling.filter_functions import (
    get_ding_filter_function,
    get_lloyd_filter_function,
)

import matplotlib.pyplot as plt
import numpy as np


def evaluate_and_fft(model: callable, times: list):
    y = np.array([model(t) for t in times])
    y[np.isnan(y)] = np.nanmax(y)
    y /= np.max(y)
    fy = np.abs(np.fft.fft(y))
    fy /= np.max(fy)
    return y, fy


def main():
    spectrum_width = 15.269206109814778
    min_gap = 1.9999999999999964

    time = 80
    time_range = np.linspace(0, time, int(1e3))

    lloyd = get_lloyd_filter_function(biga=1, beta=time / 3, tau=time / 2)
    ding = get_ding_filter_function(
        a=2.5 * spectrum_width, da=0.5 * spectrum_width, b=min_gap, db=min_gap
    )

    fig, axes = plt.subplots(nrows=2)

    lloyd_y, lloyd_fy = evaluate_and_fft(lloyd, time_range)
    ding_y, ding_fy = evaluate_and_fft(ding, time_range)

    axes[0].plot(time_range, lloyd_y, label="lloyd")
    axes[0].plot(time_range, ding_y, label="ding")
    axes[0].legend()
    axes[0].set_xlabel("t")

    axes[1].plot(time_range, lloyd_fy, label="lloyd")
    axes[1].plot(time_range, ding_fy, label="ding")
    axes[1].legend()
    axes[1].set_xlabel("$\omega$")

    plt.show()


if __name__ == "__main__":
    main()
