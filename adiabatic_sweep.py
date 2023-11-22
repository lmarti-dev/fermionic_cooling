from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
import numpy as np
from cirq import fidelity, PauliSum
from fauvqe.utilities import jw_get_true_ground_state_at_particle_number
from scipy.sparse import csc_matrix
from openfermion import FermionOperator, get_sparse_operator


def fermion_to_dense(ham: FermionOperator):
    return get_sparse_operator(ham).todense()


def ham_to_sparse(ham):
    if isinstance(ham, FermionOperator):
        return get_sparse_operator(ham)
    elif isinstance(ham, csc_matrix):
        return ham
    elif isinstance(ham, PauliSum):
        raise ValueError("can't convert PauliSum to csc_matrix")
    return ham


def get_sweep_hamiltonian(
    ham_start: np.ndarray, ham_stop: np.ndarray, sparse: bool = False
):
    if sparse:
        ham_start = ham_to_sparse(ham_start)
        ham_stop = ham_to_sparse(ham_stop)

    def ham(t):
        return (1 - t) * ham_start + t * ham_stop

    return ham


def get_instantaneous_ground_states(
    sweep_hamiltonian: callable, n_steps: int, n_electrons: list
):
    ground_states = []
    ground_energies = []
    for step in range(n_steps + 1):
        sparse_operator = csc_matrix(sweep_hamiltonian(step / n_steps))
        ground_energy, ground_state = jw_get_true_ground_state_at_particle_number(
            sparse_operator=sparse_operator, particle_number=n_electrons
        )
        ground_states.append(ground_state)
        ground_energies.append(ground_energy)

    return ground_energies, ground_states


def get_trotterized_sweep_unitaries(
    sweep_hamiltonian: callable, n_steps: int, total_time: float
):
    unitaries = []
    for step in range(n_steps + 1):
        unitary = expm(-1j * (total_time / n_steps) * sweep_hamiltonian(step / n_steps))
        unitaries.append(unitary)
    return unitaries


def run_sweep(
    initial_state: np.ndarray,
    ham_start: np.ndarray,
    ham_stop: np.ndarray,
    n_electrons: list,
    n_steps: int,
    total_time: float,
):
    # set state to initial value
    state = initial_state

    # basic stuff
    n_qubits = int(np.log2(len(state)))
    qid_shape = (2,) * n_qubits

    sweep_hamiltonian = get_sweep_hamiltonian(ham_start=ham_start, ham_stop=ham_stop)

    unitaries = get_trotterized_sweep_unitaries(
        sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, total_time=total_time
    )

    _, ground_states = get_instantaneous_ground_states(
        sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, n_electrons=n_electrons
    )

    # compute fidelities to final ground state and instant.
    fidelities = []
    instant_fidelities = []

    initial_fid = fidelity(state, ground_states[-1], qid_shape=qid_shape)
    initial_instant_fid = fidelity(state, ground_states[0], qid_shape=qid_shape)

    fidelities.append(initial_fid)
    instant_fidelities.append(initial_instant_fid)

    print(f"init. fid: {initial_fid:.4f} init. inst. fid: {initial_instant_fid:.4f}")
    for ind, unitary in enumerate(unitaries):
        state = np.matmul(unitary, state)

        fid = fidelity(state, ground_states[-1], qid_shape=qid_shape)
        instant_fid = fidelity(state, ground_states[ind], qid_shape=qid_shape)

        fidelities.append(fid)
        instant_fidelities.append(instant_fid)

        print(f"step {ind}: fid: {fid:.4f} init. inst. fid: {instant_fid:.4f}\r")
    return fidelities, instant_fidelities


def run_sparse_sweep():
    pass
