import cirq
import numpy as np
from cooling_utils import ndarray_to_psum


# This file contains a lot of legos to help with the cooling sims,
# for example common environment hamiltonians, sweeps, and couplers
# there's also the gap ansatz, which should be renamed
# the gap ansatz is the estimation of how omega should vary
# depending on the environment energy


def get_cheat_coupler(sys_eig_states, env_eig_states, qubits, to_psum: bool = False):
    coupler = 0
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for k in range(1, sys_eig_states.shape[1]):
        coupler += np.kron(
            np.outer(sys_eig_states[:, 0], np.conjugate(sys_eig_states[:, k])),
            env_up,
        )
    if to_psum:
        return ndarray_to_psum(
            coupler + np.conjugate(np.transpose(coupler)), qubits=qubits
        )
    else:
        return coupler + np.conjugate(np.transpose(coupler))


def get_log_sweep(spectrum_width: np.ndarray, n_steps: int, n_rep: int = 1):
    return np.tile(
        spectrum_width * (np.logspace(start=0, stop=-5, base=10, num=n_steps)), n_rep
    )


def get_cheat_sweep(spectrum: np.ndarray, n_steps: int = None):
    res = []
    if n_steps is None:
        n_rep = 1
    else:
        n_rep = int(n_steps / (len(spectrum) - 1))
    for k in range(len(spectrum) - 1, 0, -1):
        res.append(spectrum[k] - spectrum[0])
    return np.tile(np.array(res), n_rep)


def get_lin_sweep(spectrum: np.ndarray, n_steps: int):
    min_gap = sorted(np.abs(np.diff(spectrum)))[0]
    spectrum_width = max(spectrum) - min(spectrum)
    return np.linspace(start=spectrum_width, stop=min_gap, num=n_steps)


def get_Z_env(n_qubits):
    # environment stuff
    env_qubits = cirq.LineQubit.range(n_qubits)
    n_env_qubits = len(env_qubits)
    env_ham = -sum((cirq.Z(q) for q in env_qubits)) / 2 + n_env_qubits / 2
    env_ground_state = np.zeros((2**n_env_qubits))
    env_ground_state[0] = 1
    env_matrix = env_ham.matrix(qubits=env_qubits)
    env_energies, env_eig_states = np.linalg.eigh(env_matrix)
    return env_qubits, env_ground_state, env_ham, env_energies, env_eig_states


def get_YY_coupler(
    sys_qubits: list[cirq.Qid], env_qubits: list[cirq.Qid], n_sys_qubits: int
):
    return sum(
        [cirq.Y(sys_qubits[k]) * cirq.Y(env_qubits[k]) for k in range(n_sys_qubits)]
    )


def get_ZY_coupler(sys_qubits, env_qubits):
    n_sys_qubits = len(sys_qubits)
    n_env_qubits = len(env_qubits)
    return sum(
        [
            cirq.Z(sys_qubits[k]) * cirq.Y(env_qubits[k % n_env_qubits])
            for k in range(n_sys_qubits)
        ]
    )


def get_moving_ZY_coupler_list(sys_qubits, env_qubit):
    n_sys_qubits = len(sys_qubits)
    return list(
        cirq.Z(sys_qubits[k]) * cirq.Y(env_qubit[0]) for k in range(n_sys_qubits)
    )


def control_function(
    omega: float,
    t_fridge: float,
    beta: float = 1,
    mu: float = 1,
    c: float = 1e-5,
    f: callable = None,
):
    """Creates an ansatz for the derivative of omega, the strength of the environment. This is all control theory stuff, and I'm too dumb to get it

    Args:
        omega (float): the current value of omega
        beta (float): the general prefactor
        mu (float): the power of t_fridge
        c (float): the stabilisation variable/minimal temp
        t_fridge (float): the energy expectation of the environment
        f (callable, optional): the function wrapping omega. Defaults to identity.

    Returns:
        float: absolute value of the gradient ansatz
    """
    if f is None:
        f = lambda x: 1
    return abs(beta * f(omega) / ((t_fridge / omega) ** mu + c))
    # return abs(beta * f(omega) * np.exp(-((t_fridge * c) ** mu)))
