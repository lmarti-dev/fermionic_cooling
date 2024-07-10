from cooler_class import Cooler
from qutlet.models import FermiHubbardModel
from building_blocks import get_cheat_coupler_list, get_Z_env
import numpy as np
import matplotlib.pyplot as plt

n_electrons = [2, 2]
model = FermiHubbardModel(
    lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
)
sys_qubits = model.qubits
n_qubits = len(sys_qubits)
non_interacting_model = model.non_interacting_model.fock_hamiltonian


sys_eig_states = np.eye(N=2**n_qubits, M=2**n_qubits, dtype=complex)

sys_ground_state = sys_eig_states[:, 0]
sys_initial_state = sys_eig_states[:, -1]

n_env_qubits = 1
env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
    n_qubits=n_env_qubits
)

couplers = get_cheat_coupler_list(
    sys_eig_states=sys_eig_states,
    env_eig_states=env_eig_states,
    qubits=sys_qubits + env_qubits,
    gs_indices=(0,),
    noise=0,
)  # Interaction only on Qubit 0?

cooler = Cooler(
    sys_hamiltonian=model.hamiltonian,
    n_electrons=n_electrons,
    sys_qubits=model.qubits,
    sys_ground_state=sys_ground_state,
    sys_initial_state=sys_initial_state,
    env_hamiltonian=env_ham,
    env_qubits=env_qubits,
    env_ground_state=env_ground_state,
    sys_env_coupler_data=couplers,
    verbosity=5,
)

n_steps = 100
max_omega = 8.22
cooler.plot_controlled_cooling(
    fidelities=np.sort(np.random.rand(1, n_steps), axis=1),
    omegas=np.reshape(np.linspace(max_omega, 0.1, n_steps), (1, n_steps)),
    env_energies=np.random.rand(1, n_steps),
    eigenspectrums=max_omega * np.random.rand(1, 36),
)

plt.show()
