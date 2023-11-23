import sys

# tsk tsk
sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from coolerClass import Cooler

from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_Z_env,
    get_cheat_coupler_list,
)

from utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
)
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
import matplotlib.pyplot as plt


from data_manager import ExperimentDataManager
from scipy.ndimage.filters import gaussian_filter1d


def probe_times(
    edm: ExperimentDataManager,
    cooler: Cooler,
    alphas: np.ndarray,
    sweep_values: np.ndarray,
):
    (
        fidelities,
        energies,
        final_sys_density_matrix,
        env_energy_dynamics,
    ) = cooler.probe_evolution_times(
        alphas=alphas,
        sweep_values=sweep_values,
        N_slices=50,
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    for ind, tup in enumerate(env_energy_dynamics):
        time = np.array(tup[0]) * (alphas[ind] / np.pi)
        env_energy = np.array(tup[1]) / sweep_values[ind]
        ysmoothed = gaussian_filter1d(env_energy, sigma=2)
        ax.plot(time, ysmoothed, label=f"$\Delta E:{sweep_values[ind]:.3f}$")
    # ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel("env_energy")
    jobj = {
        "fidelities": fidelities,
        "energies": energies,
        "env_energy_dynamics": env_energy_dynamics,
    }
    edm.save_dict_to_experiment(jobj=jobj)
    edm.save_figure(fig)
    plt.show()


def __main__(args):
    data_folder = "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/data"

    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        data_folder=data_folder,
        experiment_name="cooling_cheat_both_gs",
        dry_run=dry_run,
    )
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 2]
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
    # coupler
    # coupler = cirq.Y(sys_qubits[0]) * cirq.Y(env_qubits[0])  # Interaction only on Qubit 0?
    # coupler = get_cheat_coupler(
    #     sys_eig_states=sys_eig_states,
    #     env_eig_states=env_eig_states,
    #     qubits=sys_qubits + env_qubits,
    #     gs_indices=(0,),
    # )  # Interaction only on Qubit 0?
    couplers = get_cheat_coupler_list(
        sys_eig_states=sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = sorted(np.abs(np.diff(sys_eig_energies)))[0]

    n_steps = 50
    # sweep_values = get_log_sweep(spectrum_width, n_steps)
    sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
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

    # probe_times(edm, cooler, alphas, sweep_values)

    fidelities, energies, final_sys_density_matrix = cooler.zip_cool(
        alphas=alphas,
        evolution_times=evolution_times,
        sweep_values=sweep_values,
    )

    jobj = {
        "fidelities": fidelities,
        "energies": energies,
        # "final_sys_density_matrix": final_sys_density_matrix,
    }
    edm.save_dict_to_experiment(jobj=jobj)

    print("Final Fidelity: {}".format(fidelities[-1]))

    fids = state_fidelity_to_eigenstates(final_sys_density_matrix, sys_eig_states)

    for ind, (fid, energy) in enumerate(zip(fids, sys_eig_energies)):
        print(f"<final state|E_{ind}>**2: {np.abs(fid):.5f} energy: {energy:.3f}")

    print(f"sum of fidelities: {sum(fids)}")

    fig = cooler.plot_generic_cooling(
        energies,
        fidelities,
        {"eigenspectrum": sys_eig_energies, "omegas": sweep_values},
        suptitle="Cooling 2$\\times$2 Fermi-Hubbard",
    )
    edm.save_figure(
        fig,
    )
    plt.show()


if __name__ == "__main__":
    __main__(sys.argv)