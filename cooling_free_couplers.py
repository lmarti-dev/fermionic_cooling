import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from coolerClass import Cooler

from cooling_building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_Z_env,
    get_cheat_coupler_list,
)

from cooling_utils import expectation_wrapper, ketbra, state_fidelity_to_eigenstates
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
import matplotlib.pyplot as plt


from data_manager import ExperimentDataManager


def get_min_gap(l):
    unique_vals = sorted(set(l))
    return abs(unique_vals[0] - unique_vals[1])


def __main__(args):
    data_folder = "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/data"

    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        data_folder=data_folder,
        experiment_name="cooling_check_noise_vs_alpha",
        notes="trying out the effect of noise on cheat couplers",
        dry_run=dry_run,
    )
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 1]
    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=True
    )
    sys_initial_state = ketbra(sys_hartree_fock)
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    sys_initial_energy = expectation_wrapper(
        model.hamiltonian, sys_initial_state, model.flattened_qubits
    )
    sys_ground_energy_exp = expectation_wrapper(
        model.hamiltonian, sys_ground_state, model.flattened_qubits
    )

    fidelity = cirq.fidelity(
        sys_initial_state,
        sys_ground_state,
        qid_shape=(2,) * (len(model.flattened_qubits)),
    )
    print("initial fidelity: {}".format(fidelity))
    print("ground energy from spectrum: {}".format(sys_ground_energy))
    print("ground energy from model: {}".format(sys_ground_energy_exp))
    print("initial energy from model: {}".format(sys_initial_energy))

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    edm.dump_some_variables(
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eigenspectrum=sys_eig_energies,
        env_eigenergies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )

    plt.rcParams.update(
        {
            "text.usetex": True,  # use inline math for ticks
            "font.family": r"Computer Modern Roman",  # use serif/main font for text elements
            "font.size": 15,
            "figure.figsize": (5, 4),
        }
    )

    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    couplers = get_cheat_coupler_list(
        sys_eig_states=free_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
        noise=0,
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(free_sys_eig_energies)

    n_steps = len(couplers)
    # sweep_values = get_log_sweep(spectrum_width, n_steps)
    sweep_values = get_cheat_sweep(free_sys_eig_energies, n_steps)
    # np.random.shuffle(sweep_values)
    # coupling strength value
    alphas = sweep_values / 100
    evolution_times = 2.5 * np.pi / (alphas)
    # evolution_time = 1e-3

    # call cool

    cooler = Cooler(
        sys_hamiltonian=model.hamiltonian,
        n_electrons=n_electrons,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
    )
    n_rep = 2

    ansatz_options = {"beta": 1e-2, "mu": 0.01, "c": 1e-2}
    weaken_coupling = 100

    start_omega = 1.01 * spectrum_width

    stop_omega = 0.5 * min_gap

    fidelities, sys_energies, omegas, env_energies = cooler.big_brain_cool(
        start_omega=start_omega,
        stop_omega=stop_omega,
        ansatz_options=ansatz_options,
        n_rep=n_rep,
        weaken_coupling=weaken_coupling,
    )

    jobj = {
        "fidelities": fidelities,
        "sys_energies": sys_energies,
    }
    edm.save_dict_to_experiment(filename=f"cooling_free", jobj=jobj)

    fig = cooler.plot_controlled_cooling(
        fidelities,
        sys_energies,
        omegas,
        env_energies,
        eigenspectrums=[
            sys_energies,
        ],
    )
    edm.save_figure(
        fig,
    )
    plt.show()


if __name__ == "__main__":
    __main__(sys.argv)
