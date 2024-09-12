import numpy as np
from openfermion import get_sparse_operator
from fermionic_cooling.utils import (
    state_fidelity_to_eigenstates,
    thermal_density_matrix_at_particle_number,
    fidelity,
)
from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from qutlet.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_mixed_state
import matplotlib.pyplot as plt
from data_manager import ExperimentDataManager
from fauplotstyle.styler import style


def plot_amplitudes_vs_beta(
    x, y, tunneling, coulomb, n_electrons, zoom: bool = False, n_steps: int = 200
):
    model = FermiHubbardModel(
        x_dimension=x,
        y_dimension=y,
        n_electrons=n_electrons,
        tunneling=tunneling,
        coulomb=coulomb,
    )
    n_qubits = len(model.qubits)

    eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    n_dims = len(eigenstates[:, 0])
    betas = np.zeros((n_steps,))

    total_fids = np.zeros((n_dims, n_steps))
    for ind, beta_power in enumerate(np.linspace(-2, 2, n_steps)):
        beta = 10**beta_power
        betas[ind] = beta
        thermal_sys_density = thermal_density_matrix_at_particle_number(
            beta=beta,
            sparse_operator=get_sparse_operator(model.fock_hamiltonian),
            particle_number=n_electrons,
            expanded=False,
        )
        fidelities = state_fidelity_to_eigenstates(
            thermal_sys_density, eigenstates=eigenstates, expanded=False
        )

        total_fids[:, ind] = fidelities

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("turbo", len(total_fids))
    if zoom:
        axins = ax.inset_axes(
            [0.1, 0.1, 0.3, 0.4],
            xlim=(2e-2, 1e-1),
            ylim=(0.0004, 0.001),
        )

    for ind, fids in enumerate(total_fids):
        ax.plot(
            betas * np.abs(eigenenergies[0]),
            np.abs(fids),
            label=rf"$|E_{{{ind}}}\rangle$",
            color=cmap(ind),
        )
        if zoom:
            axins.plot(
                betas * np.abs(eigenenergies[0]),
                np.abs(fids),
                label=rf"$|E_{{{ind}}}\rangle$",
                color=cmap(ind),
            )

    ax.set_ylabel("Amplitude")
    ax.set_xlabel(r"$\beta E_0$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-10, 2)
    if zoom:
        axins.yaxis.tick_right()
        ax.indicate_inset_zoom(axins, edgecolor="black")

    return fig


def plot_mms_fidelity_vs_beta(edm):
    fig, ax = plt.subplots()
    for x, y, n_electrons in ((1, 2, [1, 1]), (2, 2, [2, 2]), (2, 3, [3, 3])):
        model = FermiHubbardModel(
            x_dimension=x,
            y_dimension=y,
            n_electrons=n_electrons,
            tunneling=1,
            coulomb=2,
        )
        n_qubits = len(model.qubits)

        mms = spin_dicke_mixed_state(
            n_qubits=n_qubits, n_electrons=n_electrons, expanded=False
        )

        n_steps = 200

        edm.var_dump(model=model.__to_json__, n_electrons=n_electrons)

        mms_fids = np.zeros((n_steps,))
        for ind, beta_power in enumerate(np.linspace(-2, 2, n_steps)):
            beta = 10**beta_power

            thermal_sys_density = thermal_density_matrix_at_particle_number(
                beta=beta,
                sparse_operator=get_sparse_operator(model.fock_hamiltonian),
                particle_number=n_electrons,
                expanded=False,
            )
            mms_fids[ind] = fidelity(thermal_sys_density, mms)

        ax.plot(
            range(len(mms_fids)),
            np.abs(mms_fids),
            label=rf"${x} \times {y}$",
        )

    ax.set_ylabel("Fidelity")
    ax.set_xlabel(r"$\beta$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    edm.save_figure(fig)
    plt.show()


if __name__ == "__main__":
    style()

    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="plot_components_vs_beta_fh22", dry_run=dry_run
    )
    fig = plot_amplitudes_vs_beta(2, 2, 1, 2)
    # plot_mms_fidelity_vs_beta(edm)
