from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_spin_correct_indices,
)
from openfermion import get_sparse_operator
import numpy as np
from cirq import fidelity

import matplotlib.pyplot as plt
from fauplotstyle.styler import use_style
from data_manager import ExperimentDataManager


def plot_free_state_overlap():
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_qubits = len(model.flattened_qubits)
    n_electrons = [2, 2]
    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)

    matrix = get_sparse_operator(model.fock_hamiltonian, n_qubits=n_qubits).toarray()

    _, eig_states = np.linalg.eigh(matrix)

    fig, ax = plt.subplots()

    n_states_subspace = free_sys_eig_states.shape[1]
    n_states_total = eig_states.shape[1]

    fids = np.zeros((n_states_subspace, n_states_total, 2))

    correct_indices = list(
        sorted(jw_spin_correct_indices(n_electrons=n_electrons, n_qubits=n_qubits))
    )

    cmap = plt.get_cmap("faucmap", n_states_subspace)

    for ind1, indc in enumerate(correct_indices):
        free_state = free_sys_eig_states[:, ind1]
        for ind2 in range(n_states_total):
            # honestly fuck these shenanigans
            eig_state = eig_states[:, ind2]
            fids[ind1, ind2, :] = (
                indc,
                fidelity(free_state, eig_state, qid_shape=(2,) * n_qubits),
            )
        ax.plot(fids[ind1, :, 0], fids[ind1, :, 1], color=cmap(ind1))
    for eig_ind, indc in enumerate(correct_indices):
        max_ind = np.max(fids[:, indc, 1])
        ax.annotate(
            rf"$|E_{{{eig_ind+1}}}\rangle$",
            xytext=(indc, 0.5),
            xy=(indc, 0),
            arrowprops=dict(arrowstyle="->"),
            ha="center",
            va="center",
        )
    ax.set_xlabel("Eigenstate index")
    ax.set_ylabel("Fidelity")

    return fig


if __name__ == "__main__":
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="ideal_coupler_overlap", dry_run=dry_run
    )
    use_style()
    fig = plot_free_state_overlap()
    edm.save_figure(fig, fig_shape="page-wide")

    plt.show()
