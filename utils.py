import itertools
import multiprocessing as mp
import time
from typing import Iterable, Iterator

import cirq
import numpy as np


# TODO: replace functions with numpy equivalents
try:
    import cupy as cp
    from cupyx.scipy.linalg import expm as cupy_expm

    NO_CUPY = False
except ImportError:
    NO_CUPY = True
from openfermion import FermionOperator, get_sparse_operator, jw_hartree_fock_state
from scipy.linalg import sqrtm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, expm, expm_multiply

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import (
    chained_matrix_multiplication,
    flatten,
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
    normalize_vec,
    spin_dicke_state,
)

# this file contains general utils for the cooling.
# Some of them are simply convenience functions
# Some of them are a bit less trivial


def get_closest_noninteracting_degenerate_ground_state(
    model: FermiHubbardModel, n_qubits: int, Nf: list
):
    sparse_fermion_operator = get_sparse_operator(
        model.fock_hamiltonian, n_qubits=n_qubits
    )
    ground_energy, ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_fermion_operator, Nf
    )
    sparse_quadratic_fermion_operator = get_sparse_operator(
        model.non_interacting_model.fock_hamiltonian,
        n_qubits=n_qubits,
    )
    slater_eigenenergies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_quadratic_fermion_operator, Nf, expanded=True
    )

    return get_closest_degenerate_ground_state(
        ref_state=ground_state,
        comp_energies=slater_eigenenergies,
        comp_states=slater_eigenstates,
    )


def get_closest_degenerate_ground_state(
    ref_state: np.ndarray,
    comp_energies: np.ndarray,
    comp_states: np.ndarray,
):
    ix = np.argsort(comp_energies)
    comp_states = comp_states[:, ix]
    comp_energies = comp_energies[ix]
    comp_ground_energy = comp_energies[0]
    degeneracy = sum(
        (np.isclose(comp_ground_energy, eigenenergy) for eigenenergy in comp_energies)
    )
    fidelities = []
    if degeneracy > 1:
        print("ground state is {}-fold degenerate".format(degeneracy))
        for ind in range(degeneracy):
            fidelities.append(cirq.fidelity(comp_states[:, ind], ref_state))
        max_ind = np.argmax(fidelities)
        print(f"degenerate fidelities: {fidelities}, max: {max_ind}")
        return comp_ground_energy, comp_states[:, max_ind], max_ind
    else:
        return comp_ground_energy, comp_states[:, 0], 0


def get_min_gap(_list: list, threshold: float = 0):
    unique_vals = sorted(set(_list))
    diff = np.array(list(set(np.abs(np.diff(unique_vals)))))
    min_gap = np.min(diff[diff > threshold])
    return min_gap


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
    return np.conj(U.T)


def time_evolve_state(ham: np.ndarray, ket: np.ndarray, t: float):
    return expm_multiply(A=-1j * t * ham, B=ket)


def time_evolve_density_matrix(
    ham: np.ndarray,
    rho: np.ndarray,
    t: float,
    method: str = "expm",
):
    if method == "expm_multiply":
        # can be extremely slow
        # Ur
        Ut_rho = expm_multiply(A=-1j * t * ham, B=rho)
        # (U(Ur)*)*
        Ut_rho_Utd = dagger(expm_multiply(A=-1j * t * ham, B=dagger(Ut_rho)))
    elif method == "expm":
        Ut = expm(-1j * t * ham)
        Ut_rho_Utd = Ut @ rho @ Ut.T.conjugate()
    elif method == "cupy":
        raise NotImplementedError
        # borken
        ham = cp.array(ham, dtype=complex)
        rho = cp.array(rho, dtype=complex)
        Ut = cupy_expm(-1j * t * ham)
        Ut_rho_Utd = cp.matmul(cp.matmul(Ut, rho), Ut.T.conjugate())
    elif method == "sparse":
        # definitely do NOT use this
        # it's very slow
        ham = csc_matrix(ham)
        rho = csc_matrix(rho)
        Ut_rho = expm_multiply(A=-1j * t * ham, B=rho)
        Ut_rho_Utd = dagger(expm_multiply(A=-1j * t * ham, B=dagger(Ut_rho)))
        Ut_rho_Utd = Ut_rho_Utd.toarray()
    elif method == "diag":
        # here we use the diag method because cupy's expm doesn't
        # work for complex matrices.. only real ones (will know)
        _D, _V = cp.linalg.eigh(cp.asarray(ham, dtype=complex))
        _dV = _V.T.conjugate()
        Ut = cp.matmul(cp.matmul(_V, cp.diag(cp.exp(-1j * t * _D))), _dV)
        Utd = Ut.T.conjugate()
        Ut_rho_Utd = cp.matmul(cp.matmul(Ut, cp.asarray(rho, dtype=complex)), Utd).get()

    if not cirq.is_hermitian(Ut_rho_Utd):
        raise ValueError("time-evolved density matrix is not hermitian")
    return Ut_rho_Utd


def get_ground_state(ham: cirq.PauliSum, qubits: Iterable[cirq.Qid]) -> np.ndarray:
    _, ground_state = eigsh(ham.matrix(qubits=qubits), k=1, which="SA")
    return ground_state


def get_list_depth(_list, depth=0):
    if isinstance(_list, (list, tuple)):
        return get_list_depth(_list[0], depth=depth + 1)
    return depth


def depth_indexing(_list, indices: Iterator):
    # get l[a][b][c][d] until indices or list depth is exhausted
    ind = next(indices, None)
    if isinstance(_list, (list, tuple)):
        if ind is None:
            raise ValueError("Indices shorter than list depth")
        return depth_indexing(_list[ind], indices)
    return _list


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
    return cirq.partial_trace(
        rho.reshape(*[2 for _ in range(2 * n_sys_qubits + 2 * n_env_qubits)]),
        range(n_sys_qubits),
    ).reshape(2**n_sys_qubits, 2**n_sys_qubits)
    # cirq is faster
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


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns the quantum fidelity between two objects, each of with being either a wavefunction (a vector) or a density matrix
    Args:
        a (np.ndarray): the first object
        b (np.ndarray): the second object
    Raises:
        ValueError: if there is a mismatch in the dimensions of the objects
        ValueError: if a tensor has more than 2 dimensions
    Returns:
        float: the fidelity between the two objects
    """
    # remove empty dimensions
    squa = np.squeeze(a)
    squb = np.squeeze(b)
    # check for dimensions mismatch
    if len(set((*squa.shape, *squb.shape))) > 1:
        raise ValueError("Dimension mismatch: {} and {}".format(squa.shape, squb.shape))
    # case for two vectors
    if len(squa.shape) == 1 and len(squb.shape) == 1:
        return np.sqrt(
            np.abs(np.dot(np.conj(squa), squb) * np.dot(np.conj(squb), squa))
        )
    else:
        # case for one matrix and one vector, or two matrices
        items = []
        for item in (squa, squb):
            if len(item.shape) == 1:
                items.append(np.outer(item, item))
            elif len(item.shape) == 2:
                items.append(item)
            else:
                raise ValueError(
                    "expected vector or matrix, got {}dimensions".format(item.shape)
                )

        items[0] = sqrtm(items[0])
        rho_sigma_rho = chained_matrix_multiplication(
            np.matmul, items[0], items[1], items[0]
        )
        final_mat = sqrtm(rho_sigma_rho)
        return np.trace(final_mat) ** 2


def state_fidelity_to_eigenstates(
    state: np.ndarray, eigenstates: np.ndarray, expanded: bool = True
):
    # eigenstates have shape N * M where M is the number of eigenstates

    fids = []

    for jj in range(eigenstates.shape[1]):
        if expanded:
            fids.append(
                cirq.fidelity(
                    state, eigenstates[:, jj], qid_shape=(2,) * int(np.log2(len(state)))
                )
            )
        else:
            # in case we have fermionic vectors which aren't 2**n
            # expanded refers to jw_ restricted spaces functions
            fids.append(fidelity(state, eigenstates[:, jj]) ** 2)
    return fids


def coupler_fidelity_to_ground_state_projectors(
    coupler: np.ndarray,
    sys_eig_states: np.ndarray,
    env_eig_states: np.ndarray,
    exponentiate: bool = True,
):
    measures = []
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for k in range(1, sys_eig_states.shape[1]):
        projector = np.kron(
            np.outer(sys_eig_states[:, 0], np.conjugate(sys_eig_states[:, k])),
            env_up,
        )
        projector += np.conjugate(np.transpose(projector))
        # for now, settle for Hilbert-Schmidt inner product instead of fidelity
        # since ZY is no density matrix...
        if exponentiate:
            exp_coupler = expm(-1j * coupler)
            exp_coupler = exp_coupler / np.trace(exp_coupler)

            exp_projector = expm(-1j * projector)
            exp_projector = exp_projector / np.trace(exp_projector)
        else:
            exp_coupler = coupler
            exp_projector = projector
        measures.append(
            np.trace(np.conjugate(np.transpose(exp_coupler)) @ exp_projector)
        )
    return measures


def print_coupler_fidelity_to_ground_state_projectors(
    coupler: np.ndarray, sys_eig_states: np.ndarray, env_eig_states: np.ndarray
):
    fidelities = coupler_fidelity_to_ground_state_projectors(
        coupler=coupler,
        sys_eig_states=sys_eig_states,
        env_eig_states=env_eig_states,
    )
    for ind, fid in enumerate(fidelities):
        print(f"|ψ₀Xψ({ind+1})|: {np.real(fid):.5f}", end=", ")
    print("\n")


def extrapolate_ground_state_non_interacting_fermi_hubbard(
    model: FermiHubbardModel, n_electrons: list, n_points: int, deg: int = 1
):
    """This function finds the ground state of n_points weakly interacting FH models,
     and extrapolate via a polyfit to the non-interacting ground state.
     Currently useless (physically speaking).


    Args:
        model (FermiHubbardModel): _description_
        n_electrons (list): _description_
        n_points (int): _description_
        deg (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    coefficients = np.zeros((n_points, 2 ** len(model.flattened_qubits)))
    interval = np.linspace(1e-8, 1e-7, n_points)
    for ind, epsilon in enumerate(interval):
        params = model.to_json_dict()["constructor_params"]
        params["coulomb"] = epsilon
        model_eps = FermiHubbardModel(**params)
        _, ground_state = jw_get_true_ground_state_at_particle_number(
            get_sparse_operator(model_eps.fock_hamiltonian), particle_number=n_electrons
        )
        coefficients[ind, :] = ground_state

    poly = np.polyfit(interval, coefficients, deg)
    ground_state_extrapolated = np.polyval(poly, 0)

    return normalize_vec(ground_state_extrapolated)


def get_slater_spectrum(model: FermiHubbardModel, n_electrons: list):
    slater_energies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    return slater_energies, slater_eigenstates


def get_onsite_spectrum(model: FermiHubbardModel, n_electrons: list):
    params = model.to_json_dict()
    params["constructor_params"]["tunneling"] = 0
    onsite_model = FermiHubbardModel(**params["constructor_params"])
    slater_energies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(onsite_model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )
    return slater_energies, slater_eigenstates


def get_dicke_state(n_qubits, n_electrons):
    dicke_state = spin_dicke_state(
        n_qubits=n_qubits, Nf=n_electrons, right_to_left=False
    )
    return dicke_state


def get_closest_slater(model: FermiHubbardModel, n_electrons: list):
    (
        slater_energy,
        slater_state,
        max_ind,
    ) = get_closest_noninteracting_degenerate_ground_state(
        model=model, n_qubits=len(model.flattened_qubits), Nf=n_electrons
    )
    return slater_energy, slater_state, max_ind


def get_hartree_fock(n_qubits: int, n_electrons: list):
    hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_qubits, n_electrons=sum(n_electrons)
    )

    return hartree_fock


def get_close_ground_state(model: FermiHubbardModel, n_electrons: list, coulomb: float):
    params = model.to_json_dict()["constructor_params"]
    params["coulomb"] = coulomb
    close_model = FermiHubbardModel(**params)
    (
        close_ground_energy,
        close_ground_state,
    ) = jw_get_true_ground_state_at_particle_number(
        sparse_operator=get_sparse_operator(close_model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    return close_ground_energy, close_ground_state


def get_extrapolated_superposition(
    model: FermiHubbardModel, n_electrons: list, coulomb: float
):
    """This creates an approximation of the interacting ground state
    for given values of coulomb in terms of the degenerate non-interacting ground states

    Args:
        model (FermiHubbardModel): _description_
        n_electrons (list): _description_
        coulomb (float): _description_

    Returns:
        state (np.array): the extrapolated ground state
    """

    # get spectrum of the non interacting model
    eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    # get indices of the degenerate gs
    indices = [
        ind
        for ind in range(len(eigenenergies))
        if np.isclose(eigenenergies[ind], eigenenergies[0])
    ]

    # get the close gs
    _, close_ground_state = get_close_ground_state(
        model=model, n_electrons=n_electrons, coulomb=coulomb
    )
    coefficients = []
    for ind in indices:
        coefficients.append(np.vdot(eigenstates[:, ind], close_ground_state))
    return eigenstates[:, indices] @ np.array(coefficients)


def thermal_density_matrix_at_particle_number(
    beta: float,
    sparse_operator: csc_matrix,
    particle_number: list,
    expanded: bool = True,
):
    eig_energies, eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=sparse_operator,
        particle_number=particle_number,
        expanded=expanded,
    )

    thermal_density = np.sum(
        [
            np.exp(-beta * eig_energies[i]) * ketbra(eig_states[:, i])
            for i in range(len(eig_energies))
        ],
        axis=0,
    )
    thermal_density /= np.trace(thermal_density)
    return thermal_density


def thermal_density_matrix(beta: float, ham: cirq.PauliSum, qubits: list[cirq.Qid]):
    eig_energies, eig_states = np.linalg.eigh(ham.matrix(qubits=qubits))

    thermal_density = np.sum(
        [
            np.exp(-beta * eig_energies[i]) * ketbra(eig_states[:, i])
            for i in range(len(eig_energies))
        ],
        axis=0,
    )
    thermal_density /= np.trace(thermal_density)
    return thermal_density
