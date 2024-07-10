import sys

sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/qutlet/")

from cooler_class import (
    get_cheat_sweep,
    expectation_wrapper,
    Cooler,
    get_cheat_coupler,
)
import cirq
import numpy as np
from qutlet import Ising


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
sys_qubits = model.qubits
n_sys_qubits = len(sys_qubits)
# sys_initial_state = np.random.rand(2 ** (n[0] * n[1]))
# sys_initial_state = sys_initial_state / np.linalg.norm(sys_initial_state)
sys_initial_state = np.eye(2 ** (n[0] * n[1])) / 2 ** (n[0] * n[1])

model.diagonalise()
sys_ground_state = model.eig_vec[0]
sys_ground_energy = model.eig_val[0]

sys_initial_energy = expectation_wrapper(
    model.hamiltonian, sys_initial_state, model.qubits
)
sys_ground_energy_exp = expectation_wrapper(
    model.hamiltonian, sys_ground_state, model.qubits
)

fidelity = cirq.fidelity(
    sys_initial_state, sys_ground_state, qid_shape=(2,) * (len(model.qubits))
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
# coupler = cirq.X(sys_qubits[0]) * cirq.X(env_qubits[0]) + cirq.Y(
#    sys_qubits[0]
# ) * cirq.Y(env_qubits[0])
coupler = get_cheat_coupler(sys_eig_states=model.eig_vec, env_eig_states=model.eig_vec)

# get environment ham sweep values
spectrum_width = max(model.eig_val) - min(model.eig_val)

n_steps = 1000
sweep_values = get_cheat_sweep(model.eig_val, n_steps)
# sweep_values = get_log_sweep(spectrum_width, n_steps)
# coupling strength value
alphas = sweep_values / 5
evolution_times = np.pi / alphas

# call cool
cooler = Cooler(
    sys_hamiltonian=model.hamiltonian,
    sys_qubits=model.qubits,
    sys_ground_state=sys_ground_state,
    sys_initial_state=sys_initial_state,
    env_hamiltonian=env_ham,
    env_qubits=env_qubits,
    env_ground_state=env_ground_state,
    sys_env_coupler=coupler,
)

fidelities, energies = cooler.loop_cool(
    alphas=alphas,
    evolution_times=evolution_times,
    sweep_values=sweep_values,
)
print("Final fidelity: {}".format(fidelities[-1]))
cooler.plot_generic_cooling(energies, fidelities)
