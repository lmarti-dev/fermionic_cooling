from qutlet.models import FermiHubbardModel
from qutlet.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_spin_correct_indices,
)
from openfermion import get_sparse_operator
import numpy as np

from fermionic_cooling.utils import (
    trace_out_sys,
    trace_out_env,
    two_tensor_partial_trace,
    subspace_energy_expectation,
    fidelity_wrapper,
)
from itertools import combinations


def get_subspace_size(n_qubits, n_electrons):
    return len(list(combinations(range(n_qubits // 2), n_electrons[0]))) * len(
        list(combinations(range(n_qubits // 2), n_electrons[1]))
    )


def expand_rho(rho, n_electrons, n_qubits):
    expanded_rho = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    idx = jw_spin_correct_indices(n_electrons=n_electrons, n_qubits=n_qubits)
    expanded_rho[np.ix_(idx, idx)] = rho
    return expanded_rho


def get_rho(n_qubits, n_electrons):
    subspace_size = get_subspace_size(n_qubits, n_electrons)
    state = np.random.rand(subspace_size) + 1j * np.random.rand(subspace_size)
    rho = np.outer(np.conjugate(state), state)
    rho += np.conjugate(rho.T)
    rho /= np.trace(rho)
    expanded_rho = expand_rho(rho, n_electrons=n_electrons, n_qubits=n_qubits)
    return rho, expanded_rho


def vanilla_expec(rho, observable, qubits):
    return np.real(
        observable.expectation_from_density_matrix(
            rho.astype("complex_"),
            qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
        )
    )


if __name__ == "__main__":

    n_electrons = [2, 2]
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
    )
    n_sys_qubits = len(model.qubits)
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    subspace_size = get_subspace_size(n_qubits=n_sys_qubits, n_electrons=n_electrons)

    rho, expanded_rho = get_rho(n_qubits=n_sys_qubits, n_electrons=n_electrons)

    env_mat = np.array([[1, 0], [0, 0]])
    n_env_qubits = 1

    expanded_tensor = np.kron(expanded_rho, env_mat)
    subspace_tensor = np.kron(rho, env_mat)

    print(
        "vanilla energy:",
        vanilla_expec(expanded_rho, model.hamiltonian, model.qubits),
    )
    print(
        "subspace energy:",
        subspace_energy_expectation(rho, sys_eig_energies, sys_eig_states),
    )

    expanded_traced_sys = trace_out_env(
        rho=expanded_tensor, n_sys_qubits=n_sys_qubits, n_env_qubits=n_env_qubits
    )
    subspace_traced_sys = expand_rho(
        two_tensor_partial_trace(subspace_tensor, dim1=36, dim2=2, trace_out="dim2"),
        n_electrons=n_electrons,
        n_qubits=n_sys_qubits,
    )

    print("trace env diff", np.sum(expanded_traced_sys - subspace_traced_sys))

    expanded_traced_env = trace_out_sys(
        rho=expanded_tensor, n_sys_qubits=n_sys_qubits, n_env_qubits=n_env_qubits
    )
    subspace_traced_env = two_tensor_partial_trace(
        subspace_tensor, dim1=subspace_size, dim2=2, trace_out="dim1"
    )

    print("trace env diff", np.sum(expanded_traced_env - subspace_traced_env))

    mu, expanded_mu = get_rho(n_qubits=n_sys_qubits, n_electrons=n_electrons)

    subspace_fid = fidelity_wrapper(mu, rho, qid_shape=None, subspace_simulation=True)
    expanded_fid = fidelity_wrapper(
        expanded_mu,
        expanded_rho,
        qid_shape=(2,) * n_sys_qubits,
        subspace_simulation=False,
    )
    print(f"subspace: {subspace_fid:.4f} expanded: {expanded_fid:.4f}")
