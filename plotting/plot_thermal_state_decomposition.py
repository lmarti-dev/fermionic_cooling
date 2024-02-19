import numpy as np
from openfermion import get_sparse_operator
from utils import (
    state_fidelity_to_eigenstates,
    thermal_density_matrix_at_particle_number,
)
from utils import fidelity

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_mixed_state
import matplotlib.pyplot as plt
from data_manager import ExperimentDataManager
import fauplotstyle


def plot_amplitudes_vs_beta(edm):
    model = FermiHubbardModel(x_dimension=1, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [1, 1]
    n_qubits = len(model.flattened_qubits)

    eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )

    n_steps = 200
    n_dims = len(eigenstates[:, 0])

    edm.dump_some_variables(
        model=model.to_json_dict()["constructor_params"], n_electrons=n_electrons
    )

    total_fids = np.zeros((n_dims, n_steps))
    for ind, beta_power in enumerate(np.linspace(-2, 2, n_steps)):
        beta = 10**beta_power
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
    for ind, fids in enumerate(total_fids):
        ax.plot(
            range(len(fids)),
            np.abs(fids),
            label=rf"$|E_{{{ind}}}\rangle$",
            color=cmap(ind),
        )

    ax.set_ylabel("Amplitude")
    ax.set_xlabel(r"$\beta$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(1e-10, 2)

    edm.save_figure(fig)
    plt.show()


def plot_mms_fidelity_vs_beta(edm):
    fig, ax = plt.subplots()
    for x, y, n_electrons in ((1, 2, [1, 1]), (2, 2, [2, 2]), (2, 3, [3, 3])):
        model = FermiHubbardModel(x_dimension=x, y_dimension=y, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)

        mms = spin_dicke_mixed_state(n_qubits=n_qubits, Nf=n_electrons, expanded=False)

        n_steps = 200

        edm.dump_some_variables(
            model=model.to_json_dict()["constructor_params"], n_electrons=n_electrons
        )

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


edm = ExperimentDataManager(experiment_name="plot_components_vs_beta")
plot_mms_fidelity_vs_beta(edm)
