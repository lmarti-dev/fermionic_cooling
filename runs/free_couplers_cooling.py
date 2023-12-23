import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

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
    get_min_gap,
)
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
import matplotlib.pyplot as plt


from data_manager import ExperimentDataManager


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="cooling_free_couplers_depolnoise_1e2",
        notes="using the noninteracting coupler",
        dry_run=dry_run,
    )
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_qubits = len(model.flattened_qubits)
    n_electrons = [2, 2]

    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )

    sys_slater_state = free_sys_eig_states[:, 0]
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=False
    )
    sys_mixed_state = np.ones(2**n_sys_qubits) / (2 ** (n_sys_qubits / 2))

    # initial state setting
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

    eig_fids = state_fidelity_to_eigenstates(
        state=sys_initial_state, eigenstates=sys_eig_states
    )
    print("Initial populations")
    for fid, sys_eig_energy in zip(eig_fids, sys_eig_energies):
        print(
            f"fid: {np.abs(fid):.4f} gap: {np.abs(sys_eig_energy-sys_eig_energies[0]):.3f}"
        )
    print(f"sum fids {sum(eig_fids)}")
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
        sys_eig_energies=sys_eig_energies,
        env_eig_energies=env_eig_energies,
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

    min_gap = get_min_gap(free_sys_eig_energies, threshold=1e-6)

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
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 1, "c": 2}
    weaken_coupling = 100

    start_omega = 1.01 * spectrum_width

    stop_omega = 0.1 * min_gap

    method = "bigbrain"

    if method == "bigbrain":
        coupler_transitions = np.abs(
            np.array(free_sys_eig_energies[1:]) - free_sys_eig_energies[0]
        )
        fidelities, sys_ev_energies, omegas, env_ev_energies = cooler.big_brain_cool(
            start_omega=start_omega,
            stop_omega=stop_omega,
            ansatz_options=ansatz_options,
            n_rep=n_rep,
            weaken_coupling=weaken_coupling,
            coupler_transitions=None,
        )

        jobj = {
            "fidelities": fidelities,
            "sys_energies": sys_ev_energies,
        }
        edm.save_dict_to_experiment(filename=f"cooling_free", jobj=jobj)

        fig = cooler.plot_controlled_cooling(
            fidelities=fidelities,
            sys_energies=sys_ev_energies,
            env_energies=env_ev_energies,
            omegas=omegas,
            weaken_coupling=weaken_coupling,
            n_qubits=n_qubits,
            eigenspectrums=[
                sys_eig_energies - sys_eig_energies[0],
            ],
        )
        edm.save_figure(
            fig,
        )
        plt.show()
    elif method == "zip":
        n_steps = len(couplers)
        sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
        alphas = sweep_values / 100
        evolution_times = 2.5 * np.pi / (alphas)
        fidelities, energies, final_sys_density_matrix = cooler.zip_cool(
            alphas=alphas,
            evolution_times=evolution_times,
            sweep_values=sweep_values,
            n_rep=10,
        )

        jobj = {
            "fidelities": fidelities,
            "energies": energies,
        }
        edm.save_dict_to_experiment(filename=f"cooling_free_couplers", jobj=jobj)

        fig = cooler.plot_generic_cooling(
            energies,
            fidelities,
            suptitle="Cooling 2$\\times$2 Fermi-Hubbard",
        )

        edm.save_figure(
            fig,
        )

        plt.show()


if __name__ == "__main__":
    __main__(sys.argv)
