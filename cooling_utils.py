import itertools
import multiprocessing as mp
import time
from typing import Iterable

import cirq
import numpy as np
from openfermion import FermionOperator
from scipy.sparse.linalg import eigsh, expm, expm_multiply

from fauvqe.utilities import flatten


# this file contains general utils for the cooling.
# Some of them are simply convenience functions
# Some of them are a bit less trivial


def get_transition_rates(eigenspectrum):
    transitions = np.array(
        list(
            gap
            for gap in np.abs(
                np.array(
                    list(
                        set(
                            flatten(
                                np.diff(
                                    np.array(
                                        list(itertools.combinations(eigenspectrum, r=2))
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if not np.isclose(gap, 0)
        )
    )
    return transitions


def mean_gap(spectrum: np.ndarray):
    return float(np.mean(np.diff(spectrum)))


def is_density_matrix(state):
    return len(state.shape) == 2


def expectation_wrapper(observable, state, qubits):
    if is_density_matrix(state):
        return np.real(
            observable.expectation_from_density_matrix(
                state.astype("complex_"),
                qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
            )
        )
    else:
        return np.real(
            observable.expectation_from_state_vector(
                state.astype("complex_"),
                qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
            )
        )


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
    # print("timing...")
    start = time.time()
    if method == "expm_multiply":
        # can be extremely slow
        Ut_rho = expm_multiply(A=-1j * t * ham, B=rho)
        Ut_rho_Utd = dagger(expm_multiply(A=-1j * t * ham, B=dagger(Ut_rho)))
    elif method == "expm":
        Ut = expm(-1j * t * ham)
        Ut_rho_Utd = Ut @ rho @ Ut.transpose().conjugate()
    end = time.time()
    # print("time evolution took: {} sec".format(end - start))
    if not cirq.is_hermitian(Ut_rho_Utd):
        raise ValueError("time-evolved density matrix is not hermitian")
    return Ut_rho_Utd


def get_ground_state(ham: cirq.PauliSum, qubits: Iterable[cirq.Qid]) -> np.ndarray:
    _, ground_state = eigsh(ham.matrix(qubits=qubits), k=1, which="SA")
    return ground_state


def two_tensors_partial_trace(rho: np.ndarray, n1: int, n2: int):
    """Compute the partial trace of a two density matrix tensor product, ie rho = rho_a otimes rho_b, tr_b(rho) = rho_a
    The density matrices are assumed to have shapes rho_a = 2**n1 x 2**n2 and rho_b = 2**n2 x 2**n2

    Args:
        rho (np.ndarray): the total matrix
        n1 (int): the number of qubits in the first matrix
        n2 (int): the number of qubits in the second matrix
    """
    traced_rho = np.zeros((2**n1, 2**n1), dtype="complex_")
    # print("traced rho shape: {} rho shape: {}".format(traced_rho.shape, rho.shape))
    for iii in range(2**n1):
        for jjj in range(2**n1):
            # take rho[i*env qubits:i*env qubits + env qubtis]
            traced_rho[iii, jjj] = np.trace(
                rho[
                    iii * (2**n2) : (iii + 1) * (2**n2),
                    jjj * (2**n2) : (jjj + 1) * (2**n2),
                ]
            )
    return traced_rho


def trace_out_env(
    rho: np.ndarray,
    n_sys_qubits: int,
    n_env_qubits: int,
):
    return two_tensors_partial_trace(rho=rho, n1=n_sys_qubits, n2=n_env_qubits)


def trace_out_sys(
    rho: np.ndarray,
    n_sys_qubits: int,
    n_env_qubits: int,
):
    return cirq.partial_trace(
        rho.reshape(*[2 for _ in range(2 * n_sys_qubits + 2 * n_env_qubits)]),
        range(n_sys_qubits, n_sys_qubits + n_env_qubits),
    ).reshape(2**n_env_qubits, 2**n_env_qubits)


def ketbra(ket: np.ndarray):
    return np.outer(ket, dagger(ket))


def has_increased(val_current: float, val_previous: float, quantname: str):
    return "{q} has {i}".format(
        q=quantname,
        i="increased" if np.real(val_current) > np.real(val_previous) else "decreased",
    )


def fermionic_spin_and_number(n_qubits):
    n_up_op = sum(
        [FermionOperator("{x}^ {x}".format(x=x)) for x in range(0, n_qubits, 2)]
    )
    n_down_op = sum(
        [FermionOperator("{x}^ {x}".format(x=x)) for x in range(1, n_qubits, 2)]
    )
    n_total_op = sum(n_up_op, n_down_op)
    return n_up_op, n_down_op, n_total_op


def pauli_string_coeff_dispatcher(data):
    return pauli_string_coeff(*data)


def pauli_string_coeff(
    mat: np.ndarray, pauli_product: list[cirq.Pauli], qubits: list[cirq.Qid]
):
    pauli_string = cirq.PauliString(*[m(q) for m, q in zip(pauli_product, qubits)])
    pauli_matrix = pauli_string.matrix(qubits=qubits)
    coeff = np.trace(mat @ pauli_matrix) / mat.shape[0]
    return coeff, pauli_string


def ndarray_to_psum(
    mat: np.ndarray,
    qubits: list[cirq.Qid] = None,
    n_jobs: int = 32,
    verbose: bool = False,
) -> cirq.PauliSum:
    if len(list(set(mat.shape))) != 1:
        raise ValueError("the matrix is not square")
    n_qubits = int(np.log2(mat.shape[0]))
    if qubits is None:
        qubits = cirq.LineQubit.range(n_qubits)
    pauli_matrices = (cirq.I, cirq.X, cirq.Y, cirq.Z)
    pauli_products = itertools.product(pauli_matrices, repeat=n_qubits)
    pauli_sum = cirq.PauliSum()
    if n_jobs > 1:
        pool = mp.Pool(n_jobs)
        results = pool.starmap(
            pauli_string_coeff,
            ((mat, pauli_product, qubits) for pauli_product in pauli_products),
        )

        for result in results:
            coeff, pauli_string = result
            pauli_sum += cirq.PauliString(pauli_string, coeff)
        pool.close()
        pool.join()
    else:
        for pauli_product in pauli_products:
            if verbose:
                print(pauli_product, coeff)
            if not np.isclose(np.abs(coeff), 0):
                pauli_sum += cirq.PauliString(pauli_string, coeff)
    return pauli_sum
