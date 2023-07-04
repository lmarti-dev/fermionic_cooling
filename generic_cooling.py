# define two Pauli Sums which are H_S (FermionicModel) and H_E (sum of Z simplest, but perhaps better choice like Heisenberg)
# and third PauliSum interaction V
# do time evolution H_S + beta * H_E + alpha + V
# 1) trace out E from the density matrix of the whole thing
# rentensor this traced out mixed state with gs of H_E
# 2) *not implemented* you could either measure E and restart if it's still in gs
# or apply gs circuit if it's not

# 1) measure energy, fidelity of density matrices with correct gs
# after every step

# since the population is distributed across all excited energy levels,
# we need to sweep the coupling in H_E and change V (V is |E_j><E_i| but we dont know it we just approximate it)
# so that it matches the cooling trnasition that we want to simulate
# it's good to know the eigenspectrum of the system (but irl we don't know)
# get max(eigenvalues)-min(eigenvalues)
# logsweep is log spaced omega sampling

import numpy as np
import cirq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply, eigsh, expm
from typing import Iterable
import time


def get_psum_qubits(psum: cirq.PauliSum) -> Iterable[cirq.Qid]:
    qubits = []
    for pstr in psum:
        qubits.extend(pstr.keys())
    return tuple(set(qubits))


def dagger(U: np.ndarray) -> np.ndarray:
    return np.transpose(np.conj(U))


def time_evolve_state(ham: np.ndarray, ket: np.ndarray, t: float):
    return expm_multiply(A=-1j * t * ham, B=ket)


def time_evolve_density_matrix(
    ham: np.ndarray, rho: np.ndarray, t: float, method: str = "expm_multiply"
):
    print("timing...")
    start = time.time()
    if method == "expm_multiply":
        # can be extremely slow
        Ut_rho = expm_multiply(A=-1j * t * ham, B=rho)
        Ut_rho_Utd = dagger(expm_multiply(A=-1j * t * ham, B=dagger(Ut_rho)))
    elif method == "expm":
        Ut = expm(-1j * t * ham)
        Ut_rho_Utd = Ut @ rho @ Ut.transpose().conjugate()
    end = time.time()
    print("time evolution took: {} sec".format(end - start))
    if not cirq.is_hermitian(Ut_rho_Utd):
        raise ValueError("time-evolved density matrix is not hermitian")
    return Ut_rho_Utd


def get_ground_state(ham: cirq.PauliSum, qubits: Iterable[cirq.Qid]) -> np.ndarray:
    _, ground_state = eigsh(ham.matrix(qubits=qubits), k=1, which="SA")
    return ground_state


def trace_out_env(
    rho: np.ndarray,
    n_sys_qubits: int,
    n_env_qubits: int,
):
    # reshaped_rho = np.reshape(rho, (2,) * 2 * (n_sys_qubits + n_env_qubits))
    # traced_rho = cirq.partial_trace(
    #     tensor=reshaped_rho, keep_indices=range(n_sys_qubits)
    # )
    # reshaped_traced_rho = np.reshape(traced_rho, (2**n_sys_qubits, 2**n_sys_qubits))
    print("tracing out environment...")

    traced_rho = np.zeros((2**n_sys_qubits, 2**n_sys_qubits), dtype="complex_")
    print("traced rho shape: {} rho shape: {}".format(traced_rho.shape, rho.shape))
    for iii in range(2**n_sys_qubits):
        for jjj in range(2**n_sys_qubits):
            # take rho[i*env qubits:i*env qubits + env qubtis]
            traced_rho[iii, jjj] = np.trace(
                rho[
                    iii * (2**n_env_qubits) : (iii + 1) * (2**n_env_qubits),
                    jjj * (2**n_env_qubits) : (jjj + 1) * (2**n_env_qubits),
                ]
            )
    return traced_rho


def ketbra(ket: np.ndarray):
    return np.outer(ket, dagger(ket))


def print_increased(val_current: float, val_previous: float, quantname: str):
    print(
        "{q} has {i}".format(
            q=quantname, i="increased" if val_current > val_previous else "decreased"
        )
    )


def cool(
    sys_hamiltonian: cirq.PauliSum,
    sys_initial_state: np.ndarray,
    sys_ground_state: np.ndarray,
    env_hamiltonian: cirq.PauliSum,
    env_ground_state: np.ndarray,
    sys_env_coupling: cirq.PauliSum,
    evolution_time: float,
    alpha: float,
    sweep_values: Iterable[float],
):
    sys_qubits = get_psum_qubits(sys_hamiltonian)
    env_qubits = get_psum_qubits(env_hamiltonian)
    if env_ground_state is None:
        env_ground_state = get_ground_state(env_hamiltonian, qubits=env_qubits)
    env_ground_density_matrix = ketbra(env_ground_state)

    total_qubits = sys_qubits + env_qubits
    initial_state = np.kron(sys_initial_state, env_ground_state)
    initial_density_matrix = ketbra(initial_state)
    if not cirq.is_hermitian(initial_density_matrix):
        raise ValueError("initial density matrix is not hermitian")
    total_density_matrix = initial_density_matrix

    fidelities = []
    energies = []

    fidelity = cirq.fidelity(
        ketbra(sys_initial_state.astype("complex_")),
        sys_ground_state,
        qid_shape=(2,) * (len(sys_qubits)),
    )
    energy = sys_hamiltonian.expectation_from_density_matrix(
        ketbra(sys_initial_state.astype("complex_")),
        qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
    )
    sys_ground_energy = np.real(
        sys_hamiltonian.expectation_from_state_vector(
            sys_ground_state,
            qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
        )
    )

    print(
        "initial fidelity to gs: {}, initial energy of traced out rho: {}, ground energy: {}".format(
            fidelity, energy, sys_ground_energy
        )
    )
    fidelities.append(fidelity)
    energies.append(energy)

    for step, env_coupling in enumerate(sweep_values):
        total_ham = (
            sys_hamiltonian + env_coupling * env_hamiltonian + alpha * sys_env_coupling
        )
        print("=== step: {} ===".format(step))
        print("env coupling value: {}".format(env_coupling))

        print("evolving...")
        total_density_matrix = time_evolve_density_matrix(
            ham=total_ham.matrix(qubits=total_qubits),
            rho=total_density_matrix,
            t=evolution_time,
            method="expm",
        )
        traced_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(sys_qubits),
            n_env_qubits=len(env_qubits),
        )
        print("computing values...")
        fidelity = cirq.fidelity(
            traced_density_matrix,
            sys_ground_state,
            qid_shape=(2,) * (len(sys_qubits)),
        )
        fidelities.append(fidelity)
        energy = sys_hamiltonian.expectation_from_density_matrix(
            traced_density_matrix,
            qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
        )
        energies.append(energy)
        print(
            "fidelity to gs: {}, energy diff of traced out rho: {}".format(
                fidelity, energy - sys_ground_energy
            )
        )

        print("retensoring...")
        total_density_matrix = np.kron(traced_density_matrix, env_ground_density_matrix)
        print_increased(fidelity, fidelities[-2], "fidelity")
        print_increased(energy, energies[-2], "energy")
    return fidelities, energies
