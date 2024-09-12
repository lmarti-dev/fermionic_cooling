import numpy as np
from openfermion import get_sparse_operator
from scipy.sparse import csc_matrix

from data_manager import ExperimentDataManager
from qutlet.models import FermiHubbardModel
from qutlet.utilities import jw_eigenspectrum_at_particle_number
from fermionic_cooling.building_blocks import get_cheat_couplers, get_Z_env

import matplotlib.pyplot as plt
from fauplotstyle.styler import style

dry_run = True
style()

model_name = "fh_slater"
edm = ExperimentDataManager(
    experiment_name=f"sweep_alpha_coolham_{model_name}", dry_run=dry_run
)
n_electrons = [2, 2]
model = FermiHubbardModel(
    lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
)
n_qubits = len(model.qubits)

if "coulomb" in model_name:
    start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
    couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
elif "slater" in model_name:
    start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
    couplers_fock_hamiltonian = start_fock_hamiltonian

# free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
#     sparse_operator=get_sparse_operator(
#         couplers_fock_hamiltonian,
#         n_qubits=len(model.qubits),
#     ),
#     particle_number=n_electrons,
#     expanded=True,
# )


sys_qubits = model.qubits
n_sys_qubits = len(sys_qubits)
slater_index = 0
sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.fock_hamiltonian,
        n_qubits=len(model.qubits),
    ),
    particle_number=n_electrons,
    expanded=True,
)

n_env_qubits = 1
env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
    n_qubits=n_env_qubits
)

total_qubits = sys_qubits + env_qubits

coupler_index = 1
gs_index = 0

edm.var_dump(
    n_electrons=n_electrons,
    model=model.__to_json__,
    coupler_index=coupler_index,
    n_sys_qubits=n_sys_qubits,
    n_env_qubits=n_env_qubits,
    total_qubits=total_qubits,
    gs_index=gs_index,
)

couplers = get_cheat_couplers(
    sys_eig_states=sys_eig_states,
    env_eig_states=env_eig_states,
    qubits=sys_qubits + env_qubits,
    gs_indices=(gs_index,),
    noise=0,
)
print("coupler done")

coupler = couplers[coupler_index]

n_steps = 100

alpha_eig_vals = np.zeros((len(sys_eig_energies) * len(env_eig_energies), n_steps))
alphas = []

for ind, alphpow in enumerate(np.linspace(-10, 0, n_steps)):
    alpha = (10.0**alphpow) * (sys_eig_energies[coupler_index] - sys_eig_energies[0])
    alphas.append(alpha)
    cooling_ham = (
        model.hamiltonian.matrix(qubits=total_qubits)
        + sys_eig_energies[coupler_index] * env_ham.matrix(qubits=total_qubits)
        + float(alpha) * coupler
    )
    cool_eig_energies, _ = jw_eigenspectrum_at_particle_number(
        sparse_operator=csc_matrix(cooling_ham),
        expanded=False,
        ignore_indices=[len(total_qubits) - 1],
        particle_number=n_electrons,
    )
    alpha_eig_vals[:, ind] = cool_eig_energies


fig, ax = plt.subplots()

cmap = plt.get_cmap("faucmap", alpha_eig_vals.shape[0])

for energy in range(alpha_eig_vals.shape[0]):
    ax.plot(
        alphas / sys_eig_energies[coupler_index] - sys_eig_energies[0],
        alpha_eig_vals[energy, :],
        color=cmap(energy),
    )

ax.set_xlabel(r"$\alpha/(E_{coupler_index}-E_0)$")
ax.set_ylabel("Energy")

edm.save_figure(fig)

plt.show()
