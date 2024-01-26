import sys

import matplotlib.pyplot as plt
import numpy as np
from building_blocks import (
    get_cheat_thermalizers,
    get_cheat_coupler_list,
    get_cheat_sweep,
    get_Z_env,
    get_matrix_coupler,
)
from thermalizer import Thermalizer
from openfermion import get_sparse_operator, jw_hartree_fock_state
from utils import (
    ketbra,
    state_fidelity_to_eigenstates,
    thermal_density_matrix_at_particle_number,
    thermal_density_matrix,
    get_min_gap,
)

from data_manager import ExperimentDataManager
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state


def __main__(args):
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="thermalizer_run",
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

    beta = 1

    thermal_env_density = thermal_density_matrix(
        beta=beta, ham=env_ham, qubits=env_qubits
    )
    thermal_sys_density = thermal_density_matrix_at_particle_number(
        beta=beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    couplers = get_cheat_thermalizers(
        sys_eig_states=sys_eig_states, env_eig_states=env_eig_states, reverse=True
    )

    ones_matrix = np.ones((sys_eig_states.shape[0], sys_eig_states.shape[0]))
    ones_matrix /= np.trace(ones_matrix)
    couplers = [get_matrix_coupler(ones_matrix, env_eig_states)]

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")

    # SWeep values are here dummy
    sweep_values = np.array(list(np.abs(np.diff(sys_eig_energies))))
    sweep_values = np.repeat(sweep_values[sweep_values > 1e-8], 10)

    alphas = sweep_values / 100
    evolution_times = np.pi / np.abs(alphas)
    # evolution_time = 1e-3

    # call cool

    thermalizer = Thermalizer(
        beta=beta,
        thermal_env_density=thermal_env_density,
        thermal_sys_density=thermal_sys_density,
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
    method = "zipcool"
    if method == "zipcool":
        (
            fidelities,
            sys_energies,
            env_energies,
            final_sys_density_matrix,
        ) = thermalizer.zip_cool(
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

        fids_initl = state_fidelity_to_eigenstates(sys_initial_state, sys_eig_states)
        fids_final = state_fidelity_to_eigenstates(
            final_sys_density_matrix, sys_eig_states
        )
        fids_thermal = state_fidelity_to_eigenstates(
            thermal_sys_density, sys_eig_states
        )

        for ind, (fid_init, fid_final, fid_thermal) in enumerate(
            zip(fids_initl, fids_final, fids_thermal)
        ):
            print(
                f"<p|{ind}>: init:{np.abs(fid_init):.3f} fin:{np.abs(fid_final):.3f} ther:{np.abs(fid_thermal):.3f}"
            )

        print(f"sum of final. fidelities: {sum(fids_final)}")
        print(f"sum of env_energies: {np.sum(env_energies)}")

        thermal_energy = np.sum(
            np.array([ene * np.exp(-beta * ene) for ene in sys_eig_energies])
        ) / np.sum(np.array([np.exp(-beta * ene) for ene in sys_eig_energies]))
        print(
            f"initial_sys_ene: {sys_energies[0]} final sys ene: {sys_energies[-1]}, thermal energy: {thermal_energy}"
        )

        fig = thermalizer.plot_generic_cooling(
            fidelities,
            initial_pops=fids_initl,
            env_energies=env_energies,
            suptitle="Thermalizing 2$\\times$2 Fermi-Hubbard",
        )
        edm.save_figure(
            fig,
        )
        plt.show()

    elif method == "bigbrain":
        ansatz_options = {"beta": 1, "mu": 1, "c": 10}
        weaken_coupling = 100

        spectrum_width = np.abs(np.max(sys_eig_energies) - np.min(sys_eig_energies))
        min_gap = get_min_gap(sys_eig_energies, threshold=1e-6)

        start_omega = 1.01 * spectrum_width

        stop_omega = 0.1 * min_gap

        n_rep = 1
        n_qubits = len(thermalizer.sys_hamiltonian.qubits)

        (
            fidelities,
            sys_ev_energies,
            omegas,
            env_ev_energies,
        ) = thermalizer.big_brain_cool(
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
        edm.save_dict_to_experiment(filename="cooling_free", jobj=jobj)

        fig = thermalizer.plot_controlled_cooling(
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


if __name__ == "__main__":
    __main__(sys.argv)
