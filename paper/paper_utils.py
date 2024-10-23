from fermionic_cooling import Cooler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from qutlet.utilities import flatten
from fau_colors import colors_dark


def controlled_cooling_load_plot(
    fig_filename,
    omegas,
    fidelities,
    env_energies,
    sys_eig_energies,
    tf_minus_val: int = None,
):
    if tf_minus_val is not None:
        new_env_energies = np.abs(np.array(env_energies[0]) - tf_minus_val)
        env_energies[0] = new_env_energies.astype(list)
    fig = Cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_energies,
        omegas=omegas,
        eigenspectrums=[sys_eig_energies - sys_eig_energies[0]],
    )
    return fig


def plot_fast_sweep_vs_m(final_fids_with, final_fids_without, max_sweep_fid=None):
    fig, ax = plt.subplots()
    ms = np.arange(0, len(final_fids_with))
    ax.plot(
        ms,
        1 - final_fids_with,
        marker="x",
        label="With fast sweep",
    )
    ax.plot(
        ms,
        1 - final_fids_without,
        marker="d",
        label="Without fast sweep",
    )

    if max_sweep_fid is not None:
        ax.hlines(
            1 - max_sweep_fid,
            ms[0],
            ms[-1],
            "r",
            label="Slow sweep",
            linestyles="dashed",
        )

    ax.set_ylabel("Final infidelity")
    ax.set_xlabel(r"$d_c$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    return fig


def plot_bogo_jw_coefficients(total_coefficients, max_pauli_strs):

    fig, ax = plt.subplots()
    markers = "xdos1^+"
    cmap = plt.get_cmap("fau_cmap", len(total_coefficients))
    for ind, coeffs in enumerate(total_coefficients):
        ax.plot(
            list(range(1, len(coeffs) + 1)),
            np.sort(np.abs(coeffs))[::-1],
            label=f"$V_{{({ind+1},0)}}: {max_pauli_strs[ind]}$",
            # marker=markers[ind % len(markers)],
            # markevery=5,
            color=cmap(ind),
        )

    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Pauli string index")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-4, 9e-2])
    ax.set_xlim([1, 3e3])
    # ax.legend(ncol=4)
    return fig


def plot_comparison_fast_sweep(
    jobj_with: dict,
    jobj_wout: dict,
    sys_eig_energies: np.ndarray,
    is_tf_log: bool = False,
):

    fig, axes = plt.subplots(nrows=2, sharex=True)

    labels = ["with initial sweep", "without initial sweep"]
    colors = [colors_dark.nat, colors_dark.wiso]

    for ind, jobj in enumerate((jobj_with, jobj_wout)):
        axes[0].plot(
            jobj["omegas"][0],
            1 - np.array(jobj["fidelities"][0]),
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

    axes[0].set_ylabel("Infidelity")
    axes[1].set_xlabel("$\omega$")
    axes[0].legend()

    return fig
