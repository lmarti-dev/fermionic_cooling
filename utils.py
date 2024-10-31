import itertools
import multiprocessing as mp
from typing import Iterable, Iterator, Union

import cirq
import numpy as np
from qutlet.utilities import fidelity_wrapper


# TODO: replace functions with numpy equivalents
try:
    import cupy as cp
    from cupyx.scipy.linalg import expm as cupy_expm

    NO_CUPY = False
except ImportError:
    NO_CUPY = True
from openfermion import (
    FermionOperator,
    get_sparse_operator,
    jw_hartree_fock_state,
    normal_ordered,
)
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, expm, expm_multiply


from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from qutlet.utilities import (
    jw_spin_correct_indices,
    flatten,
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
    normalize_vec,
    spin_dicke_state,
    jw_spin_restrict_operator,
    spin_dicke_mixed_state,
    to_bitstring,
    from_bitstring,
)

# this file contains general utils for the cooling.
# Some of them are simply convenience functions
# Some of them are a bit less trivial


def pauli_mask_to_pstr(pauli_mask: np.array, qubits):
    d = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    qubits_ind = [q.x for q in qubits]
    sorted_qubs = np.argsort(qubits_ind)

    return "".join(f"{d[pauli_mask[ind]]}_{qubits_ind[ind]}" for ind in sorted_qubs)


def get_thermal_weights(beta: float, sys_eig_energies: np.ndarray, max_k=None):
    if max_k is None:
        weights = [np.exp(-beta * x) for x in sys_eig_energies[1:]]
    elif isinstance(max_k, list):
        weights = [np.exp(-beta * x) for x in sys_eig_energies[max_k]]
    elif isinstance(max_k, int):
        weights = [np.exp(-beta * x) for x in sys_eig_energies[1 : max_k + 1]]
    return np.array(weights) / np.sum(weights)


def get_closest_noninteracting_degenerate_ground_state(
    model: FermiHubbardModel, n_qubits: int, n_electrons: list
):
    sparse_fermion_operator = get_sparse_operator(
        model.fock_hamiltonian, n_qubits=n_qubits
    )
    ground_energy, ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_fermion_operator, n_electrons
    )
    sparse_quadratic_fermion_operator = get_sparse_operator(
        model.non_interacting_model.fock_hamiltonian,
        n_qubits=n_qubits,
    )
    slater_eigenenergies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_quadratic_fermion_operator, n_electrons, expanded=True
    )

    return get_closest_degenerate_ground_state(
        ref_state=ground_state,
        comp_energies=slater_eigenenergies,
        comp_states=slater_eigenstates,
    )


def get_closest_state(
    ref_state: np.ndarray, comp_states: np.ndarray, subspace_simulation: bool = False
) -> tuple[np.ndarray, int]:
    fidelities = []
    for ind in range(comp_states.shape[1]):
        fid = fidelity_wrapper(
            comp_states[:, ind],
            ref_state,
            qid_shape=(2,) * int(np.log2(len(ref_state))),
            subspace_simulation=subspace_simulation,
        )
        if fid > 0.5:
            # if one state has more than .5 fid with ref, then it's necessarily the closest
            return comp_states[:, ind], int(ind)
        fidelities.append(fid)
    max_ind = np.argmax(fidelities)
    print(f"degenerate fidelities: {len(fidelities)}, max: {max_ind}")
    return comp_states[:, max_ind], int(max_ind)


def get_closest_degenerate_ground_state(
    ref_state: np.ndarray,
    comp_energies: np.ndarray,
    comp_states: np.ndarray,
    subspace_simulation: bool = False,
):
    idx = np.argsort(comp_energies)
    comp_states = comp_states[:, idx]
    comp_energies = comp_energies[idx]
    comp_ground_energy = comp_energies[0]
    degeneracy = sum(
        (np.isclose(comp_ground_energy, eigenenergy) for eigenenergy in comp_energies)
    )
    fidelities = []
    if degeneracy > 1:
        print("ground state is {}-fold degenerate".format(degeneracy))
        for ind in range(degeneracy):
            fidelities.append(
                fidelity_wrapper(
                    comp_states[:, ind],
                    ref_state,
                    qid_shape=(2,) * int(np.log2(len(ref_state))),
                    subspace_simulation=subspace_simulation,
                )
            )
        max_ind = np.argmax(fidelities)
        print(f"degenerate fidelities: {fidelities}, max: {max_ind}")
        return comp_ground_energy, comp_states[:, max_ind], int(max_ind)
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


def add_depol_noise(
    rho: np.ndarray,
    depol_noise: float,
    n_qubits: int,
    n_electrons: list,
    is_noise_spin_conserving: bool = False,
    expanded: bool = True,
):
    if depol_noise is None:
        # return the matrix
        return rho

    if is_noise_spin_conserving:
        rho_err = spin_dicke_mixed_state(
            n_qubits=n_qubits, n_electrons=n_electrons, expanded=expanded
        )
    else:
        rho_err = np.eye(N=len(rho)) / len(rho)

    return (1 - depol_noise) * rho + depol_noise * rho_err


def s_squared_penalty(n_qubits: int, n_electrons: list):
    s_plus = FermionOperator()
    s_minus = FermionOperator()
    for x in range(0, n_qubits, 2):
        s_plus += FermionOperator(f"{x}^ {x}")
    for x in range(1, n_qubits, 2):
        s_minus += FermionOperator(f"{x}^ {x}")
    s_squared = (n_electrons[0] - s_plus) ** 2 + (n_electrons[1] - s_minus) ** 2

    return normal_ordered(s_squared)


def mean_gap(spectrum: np.ndarray):
    return float(np.mean(np.diff(spectrum)))


def is_density_matrix(state):
    return len(state.shape) == 2


def expectation_wrapper(
    observable: Union[cirq.PauliSum, np.ndarray],
    state: np.ndarray,
    qubits: list[cirq.Qid],
):
    if is_density_matrix(state):
        if isinstance(observable, cirq.PauliSum):
            return np.real(
                observable.expectation_from_density_matrix(
                    state.astype("complex_"),
                    qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
                )
            )
        elif isinstance(observable, np.ndarray):
            return np.real(np.trace(observable @ state))
    else:
        if isinstance(observable, cirq.PauliSum):
            return np.real(
                observable.expectation_from_state_vector(
                    state.astype("complex_"),
                    qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
                )
            )
        elif isinstance(observable, np.ndarray):
            return np.real(np.vdot(state, observable @ state))
    raise ValueError(
        f"Got an incompatible observable and state: observable {type(observable)}, state: {type(state)}"
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


def two_tensor_partial_trace(rho: np.ndarray, dim1: int, dim2: int, trace_out="dim2"):
    traced_rho = np.zeros((dim1, dim1), dtype="complex_")
    # rho dim1 x dim2, dim1 x dim2
    if trace_out == "dim2":
        reshaped_array = np.array(
            [
                rho[d : d + dim2, f : f + dim2]
                for d in range(0, dim1 * dim2, dim2)
                for f in range(0, dim1 * dim2, dim2)
            ]
        )
        traced_rho = np.trace(reshaped_array, axis1=1, axis2=2)
        out_shape = (dim1, dim1)
        out_rho = np.reshape(traced_rho, out_shape)

    elif trace_out == "dim1":
        reshaped_array = np.array(
            [
                rho[d : dim1 * dim2 : dim2, f : dim1 * dim2 : dim2]
                for d in range(0, dim2)
                for f in range(0, dim2)
            ]
        )
        traced_rho = np.trace(reshaped_array, axis1=1, axis2=2)
        out_shape = (dim2, dim2)
        out_rho = np.reshape(traced_rho, out_shape)

    return out_rho


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


def print_state_fidelity_to_eigenstates(
    state: np.ndarray,
    eigenenergies: np.ndarray,
    eigenstates: np.ndarray,
    expanded: bool = True,
):
    eig_fids = state_fidelity_to_eigenstates(
        state=state,
        eigenstates=eigenstates,
        expanded=expanded,
    )
    print("Populations")
    for ind, (fid, eigenenergy) in enumerate(zip(eig_fids, eigenenergies)):
        if not np.isclose(fid, 0):
            print(
                f"E_{ind}: fid: {np.abs(fid):.4f} gap: {np.abs(eigenenergy-eigenenergies[0]):.3f}"
            )
    print(f"sum fids {sum(eig_fids)}")


def state_fidelity_to_eigenstates(
    state: np.ndarray,
    eigenstates: np.ndarray,
    expanded: bool = True,
):
    # eigenstates have shape N * M where M is the number of eigenstates

    fids = []

    for jj in range(eigenstates.shape[1]):
        fids.append(
            fidelity_wrapper(
                state,
                eigenstates[:, jj],
                qid_shape=(2,) * int(np.log2(len(state))),
                subspace_simulation=not expanded,
            )
        )
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
    coefficients = np.zeros((n_points, 2 ** len(model.qubits)))
    interval = np.linspace(1e-8, 1e-7, n_points)
    for ind, epsilon in enumerate(interval):
        params = model.__to_json__
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
    params = model.__to_json__
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
        n_qubits=n_qubits, n_electrons=n_electrons, right_to_left=False
    )
    return dicke_state


def get_closest_slater(model: FermiHubbardModel, n_electrons: list):
    (
        slater_energy,
        slater_state,
        max_ind,
    ) = get_closest_noninteracting_degenerate_ground_state(
        model=model, n_qubits=len(model.qubits), n_electrons=n_electrons
    )
    return slater_energy, slater_state, max_ind


def get_hartree_fock(n_qubits: int, n_electrons: list):
    hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_qubits, n_electrons=sum(n_electrons)
    )

    return hartree_fock


def get_close_ground_state(model: FermiHubbardModel, n_electrons: list, coulomb: float):
    params = model.__to_json__
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


def dense_subspace_hamiltonian(ham: FermionOperator, n_electrons: list, n_qubits: int):
    return jw_spin_restrict_operator(
        get_sparse_operator(ham),
        particle_number=n_electrons,
        n_qubits=n_qubits,
        right_to_left=True,
    ).toarray()


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


def subspace_energy_expectation(
    rho: np.ndarray, sys_eig_energies: np.ndarray, sys_eig_states: np.ndarray
):
    # rho is N by N and projectors M by N by N
    if len(rho.shape) == 1:
        rho_hat = sys_eig_states.T.conjugate() @ rho
        state_hat = np.outer(rho_hat.T.conjugate(), rho_hat)
    else:
        state = rho
        state_hat = sys_eig_states.T.conjugate() @ state @ sys_eig_states

    return np.real(np.diag(state_hat) @ sys_eig_energies)


def get_subspace_indices_with_env_qubits(
    n_electrons: list, n_sys_qubits: int, n_env_qubits: int
):
    indices = jw_spin_correct_indices(
        n_electrons=n_electrons, n_qubits=n_sys_qubits, right_to_left=True
    )
    out_indices = []
    env_offsets = []
    for env_ind in range(2**n_env_qubits):
        env_bitstring = to_bitstring(
            ind=env_ind, n_qubits=n_env_qubits, right_to_left=True
        )
        env_bitstring = env_bitstring + "0" * n_sys_qubits
        env_offset = from_bitstring(
            b=env_bitstring,
            n_qubits=n_sys_qubits + n_env_qubits,
            right_to_left=False,
        )
        env_offsets.append(env_offset)

    for sys_ind in indices:
        for env_offset in env_offsets:
            out_ind = sys_ind + env_offset
            out_indices.append(out_ind)
    return out_indices
