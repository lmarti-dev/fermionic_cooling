import matplotlib.pyplot as plt
import numpy as np
from building_blocks import get_cheat_coupler_list, get_cheat_sweep, get_Z_env
from coolerClass import Cooler
from openfermion import (
    get_quadratic_hamiltonian,
    get_sparse_operator,
    jw_hartree_fock_state,
)
from utils import ketbra

from data_manager import ExperimentDataManager, set_color_cycler
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from fauvqe_running_code.chemical_models.specificModel import SpecificModel

dry_run = True
edm = ExperimentDataManager("chemical_cooling", dry_run=dry_run)
set_color_cycler()


spm = SpecificModel("v3/FAU_O2_singlet_6e_4o_CASSCF")
n_electrons = spm.Nf
sys_qubits = spm.current_model.flattened_qubits
n_sys_qubits = len(sys_qubits)

sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(spm.current_model.fock_hamiltonian),
    particle_number=spm.Nf,
    expanded=True,
)
sys_ground_state = sys_eig_states[:, 0]

free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        get_quadratic_hamiltonian(
            spm.current_model.fock_hamiltonian,
            n_qubits=n_sys_qubits,
            ignore_incompatible_terms=True,
        )
    ),
    particle_number=spm.Nf,
    expanded=True,
)

sys_hartree_fock = jw_hartree_fock_state(
    n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
)

sys_initial_state = ketbra(sys_hartree_fock)

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
print("coupler done")

n_steps = len(couplers)
sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
# np.random.shuffle(sweep_values)
# coupling strength value
alphas = sweep_values / 100
evolution_times = 2.5 * np.pi / np.abs(alphas)

# call cool

cooler = Cooler(
    sys_hamiltonian=spm.current_model.hamiltonian,
    n_electrons=spm.Nf,
    sys_qubits=spm.current_model.flattened_qubits,
    sys_ground_state=sys_ground_state,
    sys_initial_state=sys_initial_state,
    env_hamiltonian=env_ham,
    env_qubits=env_qubits,
    env_ground_state=env_ground_state,
    sys_env_coupler_data=couplers,
    verbosity=5,
)

# probe_times(edm, cooler, alphas, sweep_values)

fidelities, sys_energies, env_energies, final_sys_density_matrix = cooler.zip_cool(
    alphas=alphas,
    evolution_times=evolution_times,
    sweep_values=sweep_values,
)

jobj = {
    "fidelities": fidelities,
    "sys_energies": sys_energies,
    "env_energies": env_energies,
    # "final_sys_density_matrix": final_sys_density_matrix,
}
edm.save_dict_to_experiment(jobj=jobj)

print("Final Fidelity: {}".format(fidelities[-1]))


fig = cooler.plot_default_cooling(
    omegas=sweep_values,
    fidelities=fidelities[1:],
    env_energies=env_energies[1:],
    suptitle="Cooling O$_2$ Singlet",
)
edm.save_figure(
    fig,
)
plt.show()
