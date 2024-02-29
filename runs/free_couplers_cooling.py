import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from chemical_models.specificModel import SpecificModel

from coolerClass import Cooler

from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_perturbed_sweep,
    get_Z_env,
    get_cheat_coupler_list,
)

from fermionic_cooling.utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
    get_min_gap,
)
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    spin_dicke_state,
    flatten,
)
import cirq
from openfermion import (
    get_sparse_operator,
    jw_hartree_fock_state,
    get_quadratic_hamiltonian,
)
import matplotlib.pyplot as plt


from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
import numpy as np


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="fh_new_freecouplers",
        notes="cooling this home cooked chemical ham with free couplers",
        dry_run=dry_run,
    )
    use_style()
    # model stuff

    model_name = "cooked/SingletO2_6e_8q"
    model_name = "fh"
    if model_name == "fh":
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        non_interacting_model = model.non_interacting_model.fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        non_interacting_model = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            non_interacting_model,
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

    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=False
    )
    sys_mixed_state = np.ones(2**n_sys_qubits) / (2 ** (n_sys_qubits / 2))
    sys_slater_state = free_sys_eig_states[:, 2]

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    # initial state setting
    sys_initial_state = ketbra(sys_hartree_fock)

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
        time_evolve_method="expm",
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 20, "c": 40}
    weaken_coupling = 100

    start_omega = 1.01 * spectrum_width

    stop_omega = 0.1 * min_gap

    method = "bigbrain"

    if method == "bigbrain":
        coupler_transitions = np.abs(
            np.array(free_sys_eig_energies[1:]) - free_sys_eig_energies[0]
        )
        depol_noise = 1e-5
        (
            fidelities,
            sys_ev_energies,
            omegas,
            env_ev_energies,
            final_sys_density_matrix,
        ) = cooler.big_brain_cool(
            start_omega=start_omega,
            stop_omega=stop_omega,
            ansatz_options=ansatz_options,
            n_rep=n_rep,
            weaken_coupling=weaken_coupling,
            coupler_transitions=None,
            depol_noise=depol_noise,
            is_noise_spin_conserving=True,
        )

        jobj = {
            "omegas": omegas,
            "fidelities": fidelities,
            "sys_energies": sys_ev_energies,
            "env_energies": env_ev_energies,
            "depol_noise": depol_noise,
        }
        edm.save_dict_to_experiment(filename="cooling_free", jobj=jobj)

        fig = cooler.plot_controlled_cooling(
            fidelities=fidelities,
            env_energies=env_ev_energies,
            omegas=omegas,
            eigenspectrums=[
                sys_eig_energies - sys_eig_energies[0],
            ],
        )
        edm.save_figure(
            fig,
        )
        plt.show()
    elif method == "zipcool":
        initial_pops = state_fidelity_to_eigenstates(
            state=sys_initial_state, eigenstates=sys_eig_states
        )
        n_rep = 1
        sweep_values = np.array(
            list(
                flatten(
                    get_perturbed_sweep(free_sys_eig_energies, x)
                    for x in np.linspace(2.5, 3.5, 30)
                )
            )
        )

        sweep_values = get_cheat_sweep(spectrum=sys_eig_energies)

        # sweep_values = get_cheat_sweep(sys_eig_energies, n_steps=len(sys_eig_energies))
        alphas = sweep_values / 50
        evolution_times = 2.5 * np.pi / (alphas)
        (
            fidelities,
            sys_energies,
            env_energies,
            final_sys_density_matrix,
        ) = cooler.zip_cool(
            alphas=alphas,
            evolution_times=evolution_times,
            sweep_values=sweep_values,
            n_rep=n_rep,
            fidelity_threshold=2,
        )

        jobj = {
            "fidelities": fidelities,
            "sys_energies": sys_energies,
            "env_energies": env_energies,
        }
        edm.save_dict_to_experiment(filename="cooling_free_couplers", jobj=jobj)

        fig, ax = plt.subplots()

        ax.plot(sweep_values, fidelities[1:], "k--")
        ax.set_xlabel("$\omega$")
        ax.set_ylabel("Fidelity")
        ax.invert_xaxis()

        edm.save_figure(
            fig,
        )

        plt.show()


if __name__ == "__main__":
    __main__(sys.argv)
