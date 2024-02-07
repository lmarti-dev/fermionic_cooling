import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from fau_colors import colors_dark
from openfermion import get_sparse_operator

from data_manager import ExtendedJSONDecoder, read_data_path, ExperimentDataManager
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import flatten, jw_eigenspectrum_at_particle_number


def load_json(fpath: os.PathLike):
    jobj = json.loads(
        io.open(fpath, encoding="utf8", mode="r").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


def plot_results(edm, with_adiab: str, wout_adiab: str, sys_eig_energies: np.ndarray):

    jobj_with = load_json(with_adiab)
    jobj_wout = load_json(wout_adiab)

    plt.rcParams["font.size"] = 16

    fig, axes = plt.subplots(nrows=2, sharex=True)

    labels = ["with initial sweep", "without initial sweep"]
    colors = [colors_dark.nat, colors_dark.wiso]

    for ind, jobj in enumerate((jobj_with, jobj_wout)):
        axes[0].plot(
            jobj["omegas"][0],
            jobj["fidelities"][0],
            color=colors[ind],
            linewidth=1.5,
            label=labels[ind],
        )

        axes[1].plot(
            jobj["omegas"][0],
            jobj["env_ev_energies"][0],
            color=colors[ind],
            linewidth=1.5,
        )

    all_energies = np.array(list(flatten(jobj["env_ev_energies"])))

    axes[1].vlines(
        sys_eig_energies - sys_eig_energies[0],
        ymin=0,
        ymax=np.nanmax(all_energies[np.isfinite(all_energies)]),
        linestyle="--",
        color="r",
        linewidth=1,
    )

    plt.xlim(min(jobj["omegas"][0]) * 0.9, max(jobj["omegas"][0]) * 1.1)

    axes[1].set_yscale("log")
    axes[1].invert_xaxis()

    axes[0].set_ylabel("Fidelity")
    axes[1].set_xlabel("$\omega$")
    axes[1].set_ylabel(r"$E_{fridge}$")
    axes[0].legend()

    # figname = fpath.split("\\")[-2]
    # edm.save_figure(fig=fig, filename=figname)

    edm.save_figure(fig)
    plt.show()


if __name__ == "__main__":

    edm = ExperimentDataManager(experiment_name="plot_initial_sweep_comparison")
    var_dump = load_json(
        "/home/eckstein/Desktop/projects/data/2024_02_07/cooling_with_initial_adiab_sweep_13h33/run_00000/logging/var_dump_2024_02_07_13_33_16.json"
    )
    sys_eig_energies = var_dump["sys_eig_energies"]
    dirname = "/home/eckstein/Desktop/projects/data/2024_02_07/cooling_with_initial_adiab_sweep_13h33/run_00000/data/"
    with_adiab = "cooling_free_adiabatic_2024_02_07_14_04_53.json"
    wout_adiab = "cooling_free_none_2024_02_07_14_37_28.json"
    plot_results(
        edm,
        os.path.join(dirname, with_adiab),
        os.path.join(dirname, wout_adiab),
        sys_eig_energies,
    )
