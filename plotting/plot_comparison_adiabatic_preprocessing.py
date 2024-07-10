import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from fau_colors import colors_dark

from data_manager import ExtendedJSONDecoder, ExperimentDataManager
from qutlet.utilities import flatten


def load_json(fpath: os.PathLike):
    jobj = json.loads(
        io.open(fpath, encoding="utf8", mode="r").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


def plot_comparison_fast_sweep(
    with_adiab: str,
    wout_adiab: str,
    sys_eig_energies: np.ndarray,
    is_tf_log: bool = False,
):

    jobj_with = load_json(with_adiab)
    jobj_wout = load_json(wout_adiab)

    fig, axes = plt.subplots(nrows=2, sharex=True)

    labels = ["with initial sweep", "without initial sweep"]
    colors = [colors_dark.nat, colors_dark.wiso]

    for ind, jobj in enumerate((jobj_with, jobj_wout)):
        axes[0].plot(
            jobj["omegas"][0],
            jobj["fidelities"][0],
            color=colors[ind],
            label=labels[ind],
        )

        # only for thermalization
        if is_tf_log:
            axes[1].plot(
                jobj["omegas"][0],
                np.abs(
                    np.array(jobj["env_ev_energies"][0]) - jobj["env_ev_energies"][0][0]
                ),
                color=colors[ind],
            )
            axes[1].set_yscale("log")
        else:
            axes[1].plot(
                jobj["omegas"][0],
                jobj["env_ev_energies"][0],
                color=colors[ind],
            )
        axes[1].set_ylabel(r"$E_F/\omega$")

    all_energies = np.array(list(flatten(jobj["env_ev_energies"])))

    axes[1].vlines(
        sys_eig_energies - sys_eig_energies[0],
        ymin=0,
        ymax=np.nanmax(all_energies[np.isfinite(all_energies)]),
        linestyle="--",
        color="r",
    )

    plt.xlim(min(jobj["omegas"][0]) * 0.9, max(jobj["omegas"][0]) * 1.1)

    axes[1].set_yscale("log")
    axes[1].invert_xaxis()

    axes[0].set_ylabel("Fidelity")
    axes[1].set_xlabel("$\omega$")
    axes[0].legend()

    # figname = fpath.split("\\")[-2]
    # edm.save_figure(fig=fig, filename=figname)

    return fig


if __name__ == "__main__":
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="plot_init_sweep_comparison_specific_gap", dry_run=dry_run
    )
    platform = "win"
    if platform == "linux":
        var_dump_fpath = "var_dump_2024_02_07_13_33_16.json"
        dirname = "/home/eckstein/Desktop/projects/data/2024_02_07/cooling_with_initial_adiab_sweep_13h33/run_00000/data/"
        with_adiab = "cooling_free_adiabatic_2024_02_07_14_04_53.json"
        wout_adiab = "cooling_free_none_2024_02_07_14_37_28.json"
    elif platform == "win":
        var_dump_fpath = "var_dump_2024_03_07_17_20_06.json"
        dirname = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_17h20\run_00000\data"
        with_adiab = "cooling_free_adiabatic_n_gaps_5_2024_03_07_18_07_55.json"
        wout_adiab = "cooling_free_none_n_gaps_5_2024_03_07_18_19_55.json"
    var_dump = load_json(os.path.join(dirname, "../logging", var_dump_fpath))
    sys_eig_energies = var_dump["sys_eig_energies"]
    fig = plot_comparison_fast_sweep(
        os.path.join(dirname, with_adiab),
        os.path.join(dirname, wout_adiab),
        sys_eig_energies,
    )
    edm.save_figure(fig)
    plt.show()
