from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_spin_correct_indices,
)
from openfermion import get_sparse_operator
import numpy as np


def subspace_expectation(
    rho: np.ndarray, sys_eig_energies: np.ndarray, sys_eig_states: np.ndarray
):
    # rho is N by N and projectors M by N by N
    return np.trace(
        sys_eig_states.T.conjugate() @ rho @ sys_eig_states @ np.diag(sys_eig_energies)
    )


if __name__ == "__main__":

    def vanilla_expec(rho, observable, qubits):
        return np.real(
            observable.expectation_from_density_matrix(
                rho.astype("complex_"),
                qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
            )
        )

    def expand_rho(rho, particle_number, n_qubits):
        expanded_rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        idx = jw_spin_correct_indices(n_electrons=particle_number, n_qubits=n_qubits)
        expanded_rho[np.ix_(idx, idx)] = rho
        return expanded_rho

    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 2]
    n_qubits = len(model.flattened_qubits)
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    projectors = np.einsum("ai,bi->iab", sys_eig_states, sys_eig_states)
    state = np.random.rand(36) + 1j * np.random.rand(36)
    rho = np.outer(np.conjugate(state), state)
    rho += np.conjugate(rho.T)
    rho /= np.trace(rho)
    expanded_rho = expand_rho(rho, particle_number=n_electrons, n_qubits=n_qubits)

    print(vanilla_expec(expanded_rho, model.hamiltonian, model.flattened_qubits))
    print(subspace_expectation(rho, sys_eig_energies, sys_eig_states))
