import numpy as np
from cirq import PauliSum
from openfermion import FermionOperator, get_sparse_operator
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from qutlet.utilities import (
    jw_get_true_ground_state_at_particle_number,
)

import multiprocessing as mp

from fermionic_cooling.utils import add_depol_noise, fidelity_wrapper


def fermion_to_dense(ham: FermionOperator):
    return get_sparse_operator(ham).toarray()


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
) -> callable:
    if sparse:
        ham_start = ham_to_sparse(ham_start)
        ham_stop = ham_to_sparse(ham_stop)

    def ham(t):
        return (1 - t) * ham_start + t * ham_stop

    return ham


def get_instantaneous_ground_states(
    ham_start: np.ndarray,
    ham_stop: np.ndarray,
    n_steps: int,
    n_electrons: list,
):
    sweep_hamiltonian = get_sweep_hamiltonian(ham_start=ham_start, ham_stop=ham_stop)
    for step in range(n_steps):
        sparse_operator = csc_matrix(sweep_hamiltonian(step / (n_steps - 1)))
        _, ground_state = jw_get_true_ground_state_at_particle_number(
            sparse_operator=sparse_operator, particle_number=n_electrons
        )
        yield ground_state


def get_trotterized_sweep_unitaries(
    sweep_hamiltonian: callable, n_steps: int, total_time: float
):
    for step in range(n_steps):
        unitary = expm(
            -1j * (total_time / n_steps) * sweep_hamiltonian(step / (n_steps - 1))
        )
        yield unitary


def get_trotterized_sweep_unitary(
    sweep_hamiltonian: callable, n_steps: int, total_time: float
):

    pool = mp.Pool(mp.cpu_count())

    def get_chunk(steps):
        for step in steps:
            unitary = expm(
                -1j * (total_time / n_steps) * sweep_hamiltonian(step / (n_steps - 1))
            )
            yield unitary

    def matmul_iter(iterable):
        return np.matmul(*iterable)

    result = (get_chunk([step, step - 1]) for step in range(n_steps - 1, 1, -1))
    for _ in range(int(np.log2(n_steps))):
        result = ((result[i], result[i + 1]) for i in len(result) - 1)
        print("len result", len(result))
        result = pool.map(matmul_iter, result)
    return result


def run_sweep(
    initial_state: np.ndarray,
    ham_start: np.ndarray,
    ham_stop: np.ndarray,
    final_ground_state: np.ndarray,
    instantaneous_ground_states: np.ndarray,
    n_steps: int,
    total_time: float,
    get_populations: bool = True,
    single_unitary: bool = False,
    depol_noise: float = None,
    is_noise_spin_conserving: bool = False,
    n_electrons: list = None,
    n_qubits: int = None,
    subspace_simulation: bool = False,
):
    # set state to initial value
    state = initial_state

    # basic stuff

    if not subspace_simulation:
        qid_shape = (2,) * n_qubits
    else:
        qid_shape = None

    sweep_hamiltonian = get_sweep_hamiltonian(ham_start=ham_start, ham_stop=ham_stop)

    if single_unitary:
        unitary = get_trotterized_sweep_unitary(
            sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, total_time=total_time
        )
        initial_fid = fidelity_wrapper(
            state,
            final_ground_state,
            qid_shape=qid_shape,
            subspace_simulation=subspace_simulation,
        )
        initial_instant_fid = fidelity_wrapper(
            state,
            initial_state,
            qid_shape=qid_shape,
            subspace_simulation=subspace_simulation,
        )

        state = unitary @ state

        final_fid = fidelity_wrapper(
            state,
            final_ground_state,
            qid_shape=qid_shape,
            subspace_simulation=subspace_simulation,
        )

        return (
            [initial_fid, final_fid],
            [initial_instant_fid, final_fid],
            final_ground_state,
            state,
        )
    else:
        print(f"Preparing {n_steps} unitaries")
        unitaries = get_trotterized_sweep_unitaries(
            sweep_hamiltonian=sweep_hamiltonian, n_steps=n_steps, total_time=total_time
        )

    print(f"Preparing {n_steps} instantaneous ground states")

    # compute fidelities to final ground state and instant.
    fidelities = []
    instant_fidelities = []

    initial_fid = fidelity_wrapper(
        state,
        final_ground_state,
        qid_shape=qid_shape,
        subspace_simulation=subspace_simulation,
    )
    initial_instant_fid = fidelity_wrapper(
        state,
        initial_state,
        qid_shape=qid_shape,
        subspace_simulation=subspace_simulation,
    )

    if get_populations:
        end_energies, end_spectrum = np.linalg.eigh(ham_stop)
        if subspace_simulation:
            populations = np.zeros((state.shape[0], n_steps))
        else:
            populations = np.zeros((2**n_qubits, n_steps))

    fidelities.append(initial_fid)
    instant_fidelities.append(initial_instant_fid)

    print(f"init. fid: {initial_fid:.4f} init. inst. fid: {initial_instant_fid:.4f}")
    for ind, unitary in enumerate(unitaries):
        state = np.matmul(unitary, state)
        if len(state.shape) == 2:
            # if we deal with a density csc_matrix
            state = np.matmul(state, np.conjugate(unitary.T))
            state /= np.trace(state)
        elif len(state.shape) > 2:
            raise ValueError(f"Expected state size 1 or 2, got {state.size}")
        # statevector
        else:
            state /= np.linalg.norm(state)

        if depol_noise is not None:
            state = add_depol_noise(
                rho=state,
                depol_noise=depol_noise,
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                is_noise_spin_conserving=is_noise_spin_conserving,
            )

        if get_populations:
            state_pops = []
            for ind_spec in range(end_spectrum.shape[1]):
                # just numpy being stupid with dimensions
                eig_state = np.array(end_spectrum[:, ind_spec])
                eig_state = eig_state.squeeze()
                state_pops.append(
                    fidelity_wrapper(
                        a=state,
                        b=eig_state,
                        qid_shape=qid_shape,
                        subspace_simulation=subspace_simulation,
                    )
                )
            populations[:, ind] = state_pops

        fid = fidelity_wrapper(
            state,
            final_ground_state,
            qid_shape=qid_shape,
            subspace_simulation=subspace_simulation,
        )
        fidelities.append(fid)

        instant_fid = -1
        if instantaneous_ground_states is not None:
            instant_fid = fidelity_wrapper(
                state,
                next(instantaneous_ground_states),
                qid_shape=qid_shape,
                subspace_simulation=subspace_simulation,
            )
            instant_fidelities.append(instant_fid)

        print(
            f"step {ind}: fid: {fid:.4f} init. inst. fid: {instant_fid:.4f}", end="\r"
        )
    if get_populations:
        return fidelities, instant_fidelities, final_ground_state, populations, state
    return fidelities, instant_fidelities, final_ground_state, state


def get_sweep_norms(ham_start: np.array, ham_stop: np.array):

    start_eig_vals, _ = np.linalg.eigh(ham_start)
    stop_eig_vals, _ = np.linalg.eigh(ham_stop)

    maxh = np.max((np.max(start_eig_vals), np.max(stop_eig_vals)))

    d_eig_vals, _ = np.linalg.eigh(ham_stop - ham_start)

    maxhd = np.max(d_eig_vals)

    return maxh, maxhd


def run_sparse_sweep():
    pass
