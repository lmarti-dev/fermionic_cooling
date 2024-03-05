from json_extender import ExtendedJSONDecoder
from data_manager import ExperimentDataManager
import json
import os
import matplotlib.pyplot as plt
from fauplotstyle.styler import use_style
import numpy as np
import io


def load_json(fpath):
    return json.loads(
        io.open(fpath, "r", encoding="utf8").read(), cls=ExtendedJSONDecoder
    )


def plot_combined(dirname):
    files = os.listdir(dirname)
    final_fids = np.zeros((len(files) // 2, 2))
    init_fids = np.zeros((len(files) // 2, 2))
    for f in files:
        jobj = load_json(os.path.join(dirname, f))
        if "none" in f:
            col = 1
        else:
            col = 0
        final_fids[jobj["n_gaps"] - 1, col] = jobj["fidelities"][0][-1]
        init_fids[jobj["n_gaps"] - 1, col] = jobj["fidelities"][0][0]

    fig, ax = plt.subplots()
    ax.plot(
        range(len(final_fids[:, 0])),
        final_fids[:, 0],
        marker="x",
        label="With fast sweep",
    )
    ax.plot(
        range(len(final_fids[:, 1])),
        final_fids[:, 1],
        marker="d",
        label="Without fast sweep",
    )

    ax.legend()
    return fig


if __name__ == "__main__":
    use_style()
    edm = ExperimentDataManager(experiment_name="plot_fastsweep_vs_m")
    dirname = "/home/eckstein/Desktop/projects/data/2024_02_29/cooling_with_initial_adiab_sweep_10h10/run_00000/data"
    fig = plot_combined(dirname)
    edm.save_figure(fig)
    plt.show()
