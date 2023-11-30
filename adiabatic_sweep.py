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
    sweep_hamiltonian: callable,
    n_steps: int,
    n_electrons: list,
):
    for step in range(1, n_steps + 2):
        sparse_operator = csc_matrix(sweep_hamiltonian(step / (n_steps + 1)))
        _, ground_state = jw_get_true_ground_state_at_particle_number(
            sparse_operator=sparse_operator, particle_number=n_electrons
        )
        yield ground_state


def get_trotterized_sweep_unitaries(
    sweep_hamiltonian: callable, n_steps: int, total_time: float
):
    for step in range(1, n_steps + 2):
        unitary = expm(
            -1j * (total_time / n_steps) * sweep_hamiltonian(step / (n_steps + 1))
        )
        yield unitary


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

    print(f"Preparing {n_steps} unitaries")
    unitaries = get_trotterized_sweep_unitaries(
        sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, total_time=total_time
    )

    print(f"Preparing {n_steps} instantaneous ground states")
    ground_states = get_instantaneous_ground_states(
        sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, n_electrons=n_electrons
    )
    _, final_ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_operator=csc_matrix(ham_stop),
        particle_number=n_electrons,
    )

    # compute fidelities to final ground state and instant.
    fidelities = []
    instant_fidelities = []

    initial_fid = fidelity(state, final_ground_state, qid_shape=qid_shape)
    initial_instant_fid = fidelity(state, initial_state, qid_shape=qid_shape)

    fidelities.append(initial_fid)
    instant_fidelities.append(initial_instant_fid)

    print(f"init. fid: {initial_fid:.4f} init. inst. fid: {initial_instant_fid:.4f}")
    for ind, unitary in enumerate(unitaries):
        state = np.matmul(unitary, state)
        if len(state.shape) == 2:
            # if we deal with a density csc_matrix
            state = np.matmul(state, np.conjugate(unitary.T))
        elif len(state.shape) > 2:
            raise ValueError(f"Expected state size 1 or 2, got {state.size}")
        fid = fidelity(state, final_ground_state, qid_shape=qid_shape)
        instant_fid = fidelity(state, next(ground_states), qid_shape=qid_shape)

        fidelities.append(fid)
        instant_fidelities.append(instant_fid)

        print(
            f"step {ind}: fid: {fid:.4f} init. inst. fid: {instant_fid:.4f}", end="\r"
        )
    return fidelities, instant_fidelities, final_ground_state


def run_sparse_sweep():
    pass
