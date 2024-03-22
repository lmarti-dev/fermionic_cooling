import cirq
import numpy as np
from fermionic_cooling.utils import ndarray_to_psum, get_transition_rates
from math import prod
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from openfermion import get_sparse_operator

# This file contains a lot of legos to help with the cooling sims,
# for example common environment hamiltonians, sweeps, and couplers
# there's also the gap ansatz, which should be renamed
# the gap ansatz is the estimation of how omega should vary
# depending on the environment energy


def get_matrix_coupler(mat: np.ndarray, env_eig_states: np.ndarray) -> np.ndarray:
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    coupler = np.kron(
        mat,
        env_up,
    )
    return coupler + np.conjugate(np.transpose(coupler))


def get_cheat_thermalizers(
    sys_eig_states,
    env_eig_states,
    reverse: bool = True,
    add_non_cross_terms: bool = False,
    max_k: int = None,
):
    if max_k is None:
        max_k = sys_eig_states.shape[1] - 1

    couplers = []
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for j in range(0, max_k):
        # |sys_j X sys_j+1| otimes |env_1 X env_0|, j<j+1
        sys_coupler = np.outer(
            sys_eig_states[:, j], np.conjugate(sys_eig_states[:, j + 1])
        )
        if add_non_cross_terms:
            sys_coupler += np.conjugate(np.transpose(sys_coupler))
        coupler = np.kron(
            sys_coupler,
            env_up,
        )
        coupler = coupler + np.conjugate(np.transpose(coupler))
        couplers.append(coupler)

    # bigger first to match cheat sweep
    if reverse:
        return list(reversed(couplers))
    else:
        return couplers


def get_cheat_coupler_list(
    sys_eig_states,
    env_eig_states,
    qubits: list[cirq.Qid] = None,
    to_psum: bool = False,
    gs_indices: int = (0,),
    noise: float = 0,
    max_k: int = None,
    use_pauli_x: bool = False,
):
    couplers = []
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    env_down = np.outer(env_eig_states[:, 0], np.conjugate(env_eig_states[:, 1]))
    # in case there are multiple ground states we can cool to (if I may say so)
    if max_k is None:
        max_k = sys_eig_states.shape[1]

    if isinstance(max_k, list):
        c_range = max_k
    else:
        c_range = range(max_k)

    for gs_index in gs_indices:
        k_range = [k for k in c_range if k != gs_index]
        for k in k_range:
            # |sys_0Xsys_k| O |env_1Xenv_0|
            if use_pauli_x:
                env_part = env_up + env_down
            else:
                env_part = env_up
            coupler = np.kron(
                np.outer(
                    sys_eig_states[:, gs_index], np.conjugate(sys_eig_states[:, k])
                ),
                env_part,
            )
            if noise is not None:
                noisy_coupler = np.random.rand(*coupler.shape)
                coupler = coupler + (noise * noisy_coupler)
            coupler = coupler + np.conjugate(np.transpose(coupler))
            if to_psum:
                coupler = ndarray_to_psum(coupler, qubits=qubits)
            couplers.append(coupler)

    # bigger first to match cheat sweep
    return list(reversed(couplers))


def get_cheat_coupler(
    sys_eig_states,
    env_eig_states,
    qubits: list[cirq.Qid] = None,
    to_psum: bool = False,
    gs_indices: int = (0,),
):
    coupler = 0
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for gs_index in gs_indices:
        for k in range(1, sys_eig_states.shape[1]):
            # |sys_0Xsys_k| O |env_1Xenv_0|
            coupler += np.kron(
                np.outer(
                    sys_eig_states[:, gs_index], np.conjugate(sys_eig_states[:, k])
                ),
                env_up,
            )
    if to_psum:
        return ndarray_to_psum(
            coupler + np.conjugate(np.transpose(coupler)), qubits=qubits
        )
    else:
        # C + C**dagger
        return coupler + np.conjugate(np.transpose(coupler))


def get_log_sweep(spectrum_width: np.ndarray, n_steps: int, n_rep: int = 1):
    return np.tile(
        spectrum_width * (np.logspace(start=0, stop=-5, base=10, num=n_steps)), n_rep
    )


def get_perturbed_sweep(spectrum: np.ndarray, lam: float, decimals: int = 5):
    sweep_values = []
    unique_energies = np.unique(spectrum.round(decimals=decimals))
    for ind in range(len(unique_energies)):
        degeneracy = sum(
            [1 if np.isclose(x, unique_energies[ind]) else 0 for x in spectrum]
        )
        if degeneracy > 1:
            values = (
                unique_energies[ind]
                + lam * np.linspace(0.01, 1, degeneracy)
                - unique_energies[0]
            )
            sweep_values.extend(values)
        else:
            sweep_values.append = unique_energies[ind]

    return np.array(sweep_values[::-1])


def get_cheat_sweep(
    spectrum: np.ndarray, n_steps: int = None, n_rep: int = None
) -> np.ndarray:
    """Return all the gaps in the spectrum (if they are bigger than 0) in reverse order, repeated n_rep times or
    eventually closest to n_steps.

    Args:
        spectrum (np.ndarray): the energy spectrum
        n_steps (int, optional): the number of steps. Defaults to None.
        n_rep (int, optional): the number of reps. Defaults to None.

    Returns:
        np.ndarray: the energy gaps
    """
    res = []
    if n_steps is None:
        if n_rep is None:
            n_rep = 1
        n_rep = n_rep
    elif n_rep is None:
        n_rep = 1
    else:
        n_rep = int(n_steps / (len(spectrum) - 1))
    for k in range(len(spectrum) - 1, 0, -1):
        gap = spectrum[k] - spectrum[0]
        if not np.isclose(gap, 0):
            res.append(gap)
    return np.tile(np.array(res), n_rep)


def get_all_gaps(spectrum: np.ndarray, n_rep: int):
    return np.tile(np.sort(np.array(get_transition_rates(spectrum)))[::-1], n_rep)


def get_lin_sweep(spectrum: np.ndarray, n_steps: int):
    min_gap = sorted(np.abs(np.diff(spectrum)))[0]
    spectrum_width = max(spectrum) - min(spectrum)
    return np.linspace(start=spectrum_width, stop=min_gap, num=n_steps)


def get_Z_env(n_qubits):
    # environment stuff
    env_qubits = cirq.LineQubit.range(n_qubits)
    n_env_qubits = len(env_qubits)
    env_ham = -sum((cirq.Z(q) for q in env_qubits)) / (2 * n_env_qubits) + 1 / 2
    env_ground_state = np.zeros((2**n_env_qubits))
    env_ground_state[0] = 1
    env_matrix = env_ham.matrix(qubits=env_qubits)
    env_energies, env_eig_states = np.linalg.eigh(env_matrix)
    return env_qubits, env_ground_state, env_ham, env_energies, env_eig_states


def get_YY_coupler(sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid]):
    n_sys_qubits = len(sys_qubits)
    return sum(
        [cirq.Y(sys_qubits[k]) * cirq.Y(env_qubits[k]) for k in range(n_sys_qubits)]
    )


def get_ZY_coupler(sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid]):
    n_sys_qubits = len(sys_qubits)
    n_env_qubits = len(env_qubits)
    return sum(
        [
            cirq.Z(sys_qubits[k]) * cirq.Y(env_qubits[k % n_env_qubits])
            for k in range(n_sys_qubits)
        ]
    )


def check_env_qubits(env_qubits: list[cirq.Qid], expected: int):
    if len(env_qubits) != expected:
        raise ValueError(f"Expected {expected} ancillas, got {len(env_qubits)}")


def get_moving_ZY_coupler_list(sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid]):
    n_sys_qubits = len(sys_qubits)
    n_env_qubits = len(env_qubits)
    YY_coupler = prod(list(cirq.Y(env_qubits[j]) for j in range(n_env_qubits)))
    return list(cirq.Z(sys_qubits[k]) * YY_coupler for k in range(n_sys_qubits))


def get_moving_paulipauli_coupler_list(
    sys_qubits: list[cirq.Qid],
    env_qubits: list[cirq.Qid],
    sys_pauli: cirq.Pauli,
    env_pauli: cirq.Pauli,
):
    n_sys_qubits = len(sys_qubits)
    return list(
        sys_pauli(sys_qubits[k]) * env_pauli(env_qubits[0]) for k in range(n_sys_qubits)
    )


def get_moving_ZYZY_coupler_list(
    sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid]
):
    n_sys_qubits = len(sys_qubits)
    return list(
        cirq.Z(sys_qubits[k]) * cirq.Y(env_qubits[0])
        + cirq.Z(sys_qubits[(k + n_sys_qubits - 1) % n_sys_qubits])
        * cirq.Y(env_qubits[1])
        for k in range(n_sys_qubits)
    )


def get_moving_fsim_coupler_list(
    sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid]
):
    n_sys_qubits = len(sys_qubits)
    return list(
        (
            cirq.X(sys_qubits[k]) * cirq.X(sys_qubits[k + 1])
            + cirq.Y(sys_qubits[k]) * cirq.Y(sys_qubits[k + 1])
        )
        * cirq.Y(env_qubits[0])
        for k in range(n_sys_qubits - 1)
    )


def control_function(
    omega: float,
    t_fridge: float,
    t_mean: float = None,
    beta: float = 1,
    mu: float = 10,
    c: float = 100,
    minus: float = 0,
    use_accelerate: bool = False,
    clamp: bool = True,
):
    """Creates an ansatz for the derivative of omega, the strength of the environment. This is all control theory stuff, and I'm too dumb to get it

    Args:
        omega (float): the current value of omega
        beta (float): the general prefactor
        mu (float): the power of t_fridge
        c (float): the stabilisation variable/minimal temp
        t_fridge (float): the energy expectation of the environment

    Returns:
        float: absolute value of the gradient ansatz
    """

    # get some momentum in the control function
    if t_mean is not None and use_accelerate:
        accelerate = (t_mean / t_fridge) ** (0.5)
    else:
        accelerate = 1

    # disallow values which might be due to bugs
    ctrl = (
        abs(beta / (np.exp(mu / (1 - np.log10(np.abs(t_fridge - minus) / omega))) + c))
        * accelerate
    )
    if clamp is True:
        return np.clip(ctrl, 1e-4, 1)
    return ctrl

    # return abs(beta * f(omega) * np.exp(-((t_fridge * c) ** mu)))


def get_hamiltonian_coupler(hamiltonian: cirq.PauliSum, env_qubits):
    coupler_list = []
    n_env_qubits = len(env_qubits)
    YY_coupler = prod(list(cirq.Y(env_qubits[j]) for j in range(n_env_qubits)))
    for pstr in hamiltonian:
        coupler_list.append(pstr * YY_coupler)
    return coupler_list


def get_perturbed_free_couplers(
    model: FermiHubbardModel,
    n_electrons: list,
    env_eig_states: np.ndarray,
    env_qubits: list[cirq.Qid],
    noise: float = 0,
    to_psum: bool = False,
):
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    couplers = []
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for k in range(1, sys_eig_states.shape[1]):
        # |sys_0Xsys_k| O |env_1Xenv_0|
        coupler = np.kron(
            np.outer(sys_eig_states[:, 0], np.conjugate(sys_eig_states[:, k])),
            env_up,
        )
        if abs(noise) > 0:
            noisy_coupler = np.random.rand(*coupler.shape)
            coupler = coupler + (noise * noisy_coupler)
        coupler = coupler + np.conjugate(np.transpose(coupler))
        if to_psum:
            coupler = ndarray_to_psum(
                coupler, qubits=(*model.flattened_qubits, *env_qubits)
            )
        couplers.append(coupler)

    # bigger first to match cheat sweep
    return list(reversed(couplers))
