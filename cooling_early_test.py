import sys

sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

import generic_cooling as gcool
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
from fauvqe.utilities import spin_dicke_state, qmap


def mean_gap(spectrum):
    return float(np.mean(np.diff(spectrum)))


def get_log_sweep(spectrum_width, n_steps):
    return spectrum_width * (np.logspace(start=0, stop=-2, base=10, num=n_steps))


def get_cheat_sweep(spectrum, n_steps):
    res = []
    n_rep = int(n_steps / (len(spectrum) - 1))
    for k in range(len(spectrum) - 1, 0, -1):
        for m in range(n_rep):
            res.append(spectrum[k] - spectrum[0])
    return np.array(res)


def get_lin_sweep(spectrum_width, n_steps):
    return np.linspace(start=spectrum_width, stop=min_gap, num=n_steps)


def expectation_wrapper(observable, state, qubits):
    return np.real(
        observable.expectation_from_state_vector(
            state.astype("complex_"),
            qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
        )
    )


# system stuff
model = FermiHubbardModel(x_dimension=1, y_dimension=2, tunneling=1, coulomb=2)
Nf = [1, 1]
sys_qubits = model.flattened_qubits
n_sys_qubits = len(sys_qubits)
sys_hartree_fock = jw_hartree_fock_state(n_orbitals=n_sys_qubits, n_electrons=sum(Nf))
sys_dicke = spin_dicke_state(n_qubits=n_sys_qubits, Nf=Nf, right_to_left=True)
sys_initial_state = sys_hartree_fock
sys_eigenspectrum, sys_eigenstates = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.fock_hamiltonian, n_qubits=len(model.flattened_qubits)
    ),
    particle_number=Nf,
    expanded=True,
)
sys_ground_state = sys_eigenstates[:, np.argmin(sys_eigenspectrum)]
sys_ground_energy = np.min(sys_eigenspectrum)

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
env_qubits = cirq.GridQubit.rect(n_sys_qubits, 1, top=Nf[0])
n_env_qubits = len(env_qubits)
env_ham = -sum((cirq.Z(q) for q in env_qubits))
env_ground_state = np.zeros((2**n_env_qubits))
env_ground_state[0] = 1

# coupler
coupler = cirq.X(sys_qubits[0]) * cirq.X(env_qubits[0])

# get environment ham sweep values
spectrum_width = max(sys_eigenspectrum) - min(sys_eigenspectrum)

min_gap = sorted(np.abs(np.diff(sys_eigenspectrum)))[0]

n_steps = 30
# sweep_values = get_log_sweep(spectrum_width, n_steps)
sweep_values = get_cheat_sweep(sys_eigenspectrum, n_steps)
# coupling strength value
alphas = sweep_values / 10
evolution_times = np.pi / (alphas)
# evolution_time = 1e-3

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
    evolution_time=evolution_times,
    sweep_values=sweep_values,
)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, figsize=(5, 3))

axes[0].plot(
    range(len(fidelities)),
    fidelities,
)
axes[0].set_ylabel("Fid. cool state with gs")
axes[1].plot(range(len(energies)), energies)
axes[1].set_ylabel("Ene. cool state")
axes[2].hlines(sys_eigenspectrum, xmin=-2, xmax=2)
axes[2].set_ylabel("Eigenenergies")


plt.show()
