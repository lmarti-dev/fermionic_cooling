import sys

sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

import generic_cooling as gcool
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
from fauvqe import Ising


def mean_gap(spectrum):
    return float(np.mean(np.diff(spectrum)))


def get_log_sweep(spectrum_width, n_steps):
    return spectrum_width * (np.logspace(start=0, stop=-5, base=10, num=n_steps))


def expectation_wrapper(observable, state, qubits):
    return np.real(
        observable.expectation_from_state_vector(
            state.astype("complex_"),
            qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
        )
    )


# system stuff
n = [1, 1]
model = Ising(
    "GridQubit",
    n,
    np.zeros((n[0], n[1])),
    np.zeros((n[0], n[1])),
    np.ones((n[0], n[1])),
    "Z",
)
sys_qubits = model.flattened_qubits
n_sys_qubits = len(sys_qubits)
sys_initial_state = np.random.rand(2 ** (n[0] * n[1]))
sys_initial_state = sys_initial_state / np.linalg.norm(sys_initial_state)

model.diagonalise()
sys_ground_state = model.eig_vec[0]
sys_ground_energy = model.eig_val[0]

sys_initial_energy = expectation_wrapper(
    model.hamiltonian, sys_initial_state, model.flattened_qubits
)
sys_ground_energy_exp = expectation_wrapper(
    model.hamiltonian, sys_ground_state, model.flattened_qubits
)

fidelity = cirq.fidelity(
    sys_initial_state, sys_ground_state, qid_shape=(2,) * (len(model.flattened_qubits))
)
print("initial fidelity: {}".format(fidelity))
print("ground energy from spectrum: {}".format(sys_ground_energy))
print("ground energy from model: {}".format(sys_ground_energy_exp))
print("initial energy from model: {}".format(sys_initial_energy))

# environment stuff
env_qubits = cirq.GridQubit.rect(n_sys_qubits, 1, top=n[0])
n_env_qubits = len(env_qubits)
env_ham = -sum((cirq.Z(q) for q in env_qubits))
env_ground_state = np.zeros((2**n_env_qubits))
env_ground_state[0] = 1

# coupler
coupler = cirq.X(sys_qubits[0]) * cirq.X(env_qubits[0])
print(coupler)

# get environment ham sweep values
spectrum_width = max(model.eig_val) - min(model.eig_val)

n_steps = 100
sweep_values = get_log_sweep(spectrum_width, n_steps)
# coupling strength value
alphas = sweep_values / 10
evolution_time = np.pi / alphas

# call cool
fidelities, energies = gcool.cool(
    sys_hamiltonian=model.hamiltonian,
    sys_qubits=model.flattened_qubits,
    sys_ground_state=sys_ground_state,
    sys_initial_state=sys_initial_state,
    env_hamiltonian=env_ham,
    env_qubits=env_qubits,
    env_ground_state=env_ground_state,
    sys_env_coupling=coupler,
    alpha=alphas,
    evolution_time=evolution_time,
    sweep_values=sweep_values,
)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, figsize=(5, 3))

axes[0].plot(
    range(len(fidelities)),
    fidelities,
)
axes[0].set_ylabel("Fid. cool state with gs")
axes[1].plot(range(len(energies)), energies)
axes[1].set_ylabel("Ene. cool state")

plt.show()
