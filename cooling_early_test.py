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
    return spectrum_width * (1 - np.logspace(start=-10, stop=-1, base=10, num=n_steps))


def get_lin_sweep(spectrum_width, n_steps):
    return np.linspace(start=spectrum_width, stop=min_gap, num=n_steps)


# system stuff
model = FermiHubbardModel(x_dimension=1, y_dimension=2, tunneling=1, coulomb=2)
Nf = [1, 1]
sys_qubits = model.flattened_qubits
n_sys_qubits = len(sys_qubits)
sys_hartree_fock = jw_hartree_fock_state(n_orbitals=n_sys_qubits, n_electrons=sum(Nf))
sys_dicke = spin_dicke_state(n_qubits=n_sys_qubits, Nf=Nf, right_to_left=True)
sys_initial_state = sys_dicke
sys_eigenspectrum, sys_eigenstates = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.fock_hamiltonian, n_qubits=len(model.flattened_qubits)
    ),
    particle_number=Nf,
    expanded=True,
)
sys_ground_state = sys_eigenstates[:, np.argmin(sys_eigenspectrum)]
sys_ground_energy = np.min(sys_eigenspectrum)

sys_initial_energy = np.real(
    model.hamiltonian.expectation_from_state_vector(
        sys_initial_state.astype("complex_"),
        qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
    )
)

print("ground energy from spectrum: {}".format(sys_ground_energy))
print("initial energy from model: {}".format(sys_initial_energy))


# environment stuff
env_qubits = cirq.GridQubit.rect(n_sys_qubits, 1)
n_env_qubits = len(env_qubits)
env_ham = sum((cirq.Z(q) for q in env_qubits))
env_ground_state = np.zeros((2**n_env_qubits))
env_ground_state[-1] = 1

# coupler
coupler = cirq.Z(sys_qubits[0]) * cirq.Z(env_qubits[0])


# get environment ham sweep values
spectrum_width = max(sys_eigenspectrum) - min(sys_eigenspectrum)

# coupling strength value
alpha = float(spectrum_width / 1e5)
evolution_time = alpha

min_gap = sorted(np.abs(np.diff(sys_eigenspectrum)))[0]

n_steps = 100
sweep_values = get_log_sweep(spectrum_width, n_steps)

# call cool
fidelities, energies = gcool.cool(
    sys_hamiltonian=model.hamiltonian,
    sys_ground_state=sys_ground_state,
    sys_initial_state=sys_initial_state,
    env_hamiltonian=env_ham,
    env_ground_state=env_ground_state,
    sys_env_coupling=coupler,
    alpha=alpha,
    evolution_time=evolution_time,
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
