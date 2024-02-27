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

from data_manager import ExperimentDataManager, set_color_cycler
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state, qmap


def print_thermalizing_stats(
    beta: float,
    sys_initial_state: np.ndarray,
    sys_energies: np.ndarray,
    sys_eig_states: np.ndarray,
    sys_eig_energies: np.ndarray,
    final_sys_density_matrix: np.ndarray,
    thermal_sys_density: np.ndarray,
    env_energies: np.ndarray,
    fidelities: np.ndarray,
):
    fids_initl = state_fidelity_to_eigenstates(sys_initial_state, sys_eig_states)
    fids_final = state_fidelity_to_eigenstates(final_sys_density_matrix, sys_eig_states)
    fids_thermal = state_fidelity_to_eigenstates(thermal_sys_density, sys_eig_states)

    for ind, (fid_init, fid_final, fid_thermal) in enumerate(
        zip(fids_initl, fids_final, fids_thermal)
    ):
        print(
            f"<p|{ind}>: init:{np.abs(fid_init):.3f} fin:{np.abs(fid_final):.3f} ther:{np.abs(fid_thermal):.3f}"
        )

    print(f"sum of init. fidelities: {sum(fids_initl)}")
    print(f"sum of final. fidelities: {sum(fids_final)}")
    print(f"sum of therm. fidelities: {sum(fids_thermal)}")
    print(f"sum of env_energies: {np.sum(env_energies)}")
    print("Initi Fidelity: {}".format(fidelities[-0]))
    print("Final Fidelity: {}".format(fidelities[-1]))

    thermal_energy = np.sum(
        np.array([ene * np.exp(-beta * ene) for ene in sys_eig_energies])
    ) / np.sum(np.array([np.exp(-beta * ene) for ene in sys_eig_energies]))
    print(
        f"initial_sys_ene: {sys_energies[0]:.3f} final sys ene: {sys_energies[-1]:.3f}, thermal energy: {thermal_energy:.3f}"
    )


def get_sweet_sweep(couplers: list, eig_vals: np.ndarray, n_rep: int):
    sweep_values = get_all_gaps(eig_vals, 0)
    idx = get_idx_above_thresh(sweep_values, threshold=1e-8)
    couplers = [couplers[jj] for jj in idx[0]]
    sweep_values = sweep_values[idx]
    idx = np.argsort(-sweep_values)
    couplers = [couplers[jj] for jj in idx]
    sweep_values = sweep_values[idx]
    sweep_values = np.tile(sweep_values, n_rep)
    return couplers, sweep_values


def get_idx_above_thresh(arr, threshold: float = 0) -> np.ndarray:
    return np.nonzero(np.abs(arr) > threshold)


def get_all_gaps(sys_eig_energies, threshold: float = 1e-6) -> np.ndarray:
    sweep_values = np.array(list(np.abs(np.diff(sys_eig_energies))))
    sweep_values = sweep_values[np.abs(sweep_values) > threshold]
    return sweep_values[::-1]


def get_all_eigs(model: FermiHubbardModel):
    matrix = get_sparse_operator(model.fock_hamiltonian).toarray()
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    return eig_vals, eig_vecs


def maximally_mixed_state(n_qubits):
    return np.eye(N=2**n_qubits, M=2**n_qubits) / (2**n_qubits)


def main_run(edm: ExperimentDataManager, initial_beta, target_beta):
    # whether we want to skip all saving data

    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    sys_qubits = model.flattened_qubits

    # define inverse temp

    env_beta = 1 * target_beta

    n_sys_qubits = len(sys_qubits)
    n_env_qubits = 1
    n_total_qubits = n_sys_qubits + n_env_qubits

    n_electrons = [2, 2]
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=True
    )

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    # renormalize target beta with ground state
    target_beta = target_beta / np.abs(sys_eig_energies[0])

    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    thermal_env_density = thermal_density_matrix(
        beta=env_beta, ham=env_ham, qubits=env_qubits
    )

    thermal_sys_density = thermal_density_matrix_at_particle_number(
        beta=target_beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    # Important part of the script where things that matter happen

    thermal_sys_initial_density = thermal_density_matrix_at_particle_number(
        beta=initial_beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )

    sys_initial_state = thermal_sys_initial_density

    couplers = get_cheat_thermalizers(
        sys_eig_states=free_sys_eig_states,
        env_eig_states=env_eig_states,
        add_non_cross_terms=True,
    )

    n_rep = 10

    spectrum_width = np.abs(np.max(sys_eig_energies) - np.min(sys_eig_energies))
    min_gap = get_min_gap(sys_eig_energies, threshold=1e-8)
    sweep_values = np.tile(np.linspace(min_gap, spectrum_width, 40), n_rep)[::-1]
    # sweep_values = get_cheat_sweep(spectrum=sys_eig_energies)
    alphas = sweep_values / 100
    evolution_times = np.pi / np.abs(alphas)

    thermalizer = Thermalizer(
        beta=target_beta,
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
        time_evolve_method="diag",
    )
    edm.dump_some_variables(
        initial_beta=initial_beta,
        target_beta=target_beta,
        env_beta=env_beta,
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eigenspectrum=sys_eig_energies,
        env_eigenergies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )

    # probe_times(edm, cooler, alphas, sweep_values)
    method = "bigbrain"
    if method == "zipcool":
        (
            fidelities,
            sys_energies,
            env_energies,
            final_sys_density_matrix,
        ) = thermalizer.loop_cool(
            alphas=alphas,
            evolution_times=evolution_times,
            sweep_values=sweep_values,
            pooling_method="max",
        )

        jobj = {
            "fidelities": fidelities,
            "sys_energies": sys_energies,
            "env_energies": env_energies,
            # "final_sys_density_matrix": final_sys_density_matrix,
        }
        edm.save_dict_to_experiment(jobj=jobj)

        print_thermalizing_stats(
            target_beta,
            sys_initial_state,
            sys_energies,
            sys_eig_states,
            sys_eig_energies,
            final_sys_density_matrix,
            thermal_sys_density,
            env_energies,
            fidelities,
        )

        fig = thermalizer.plot_default_cooling(
            omegas=sweep_values,
            fidelities=fidelities[1:],
            env_energies=env_energies[1:],
            suptitle=f"Thermalizing {model.x_dimension}$\\times${model.y_dimension} Fermi-Hubbard",
            n_rep=n_rep,
        )
        edm.save_figure(
            fig,
        )
        # plt.show()

    elif method == "bigbrain":
        ansatz_options = {
            "beta": 1,
            "mu": 20,
            "c": 40,
            "minus": env_ham.expectation_from_density_matrix(
                thermal_env_density, qubit_map={k: v for v, k in enumerate(env_qubits)}
            ),
        }
        weaken_coupling = 100

        start_omega = 1.01 * spectrum_width

        stop_omega = 0.1 * min_gap

        n_rep = 1
        n_qubits = len(thermalizer.sys_hamiltonian.qubits)

        (
            fidelities,
            sys_ev_energies,
            omegas,
            env_ev_energies,
            final_sys_density_matrix,
        ) = thermalizer.big_brain_cool(
            start_omega=start_omega,
            stop_omega=stop_omega,
            ansatz_options=ansatz_options,
            n_rep=n_rep,
            weaken_coupling=weaken_coupling,
            coupler_transitions=None,
        )

        jobj = {
            "omegas": omegas,
            "fidelities": fidelities,
            "sys_energies": sys_ev_energies,
            "env_energies": env_ev_energies,
        }
        edm.save_dict_to_experiment(filename="cooling_free", jobj=jobj)

        fig = thermalizer.plot_controlled_cooling(
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
        # plt.show()


def loop_over_betas():
    dry_run = False

    for initial_beta in (10,):
        edm = ExperimentDataManager(
            experiment_name=f"compare_thermalizers_initb_{initial_beta}",
            dry_run=dry_run,
        )

        n_steps = 20
        for ind, bpow in enumerate(np.linspace(-1, 1.7, n_steps)):
            target_beta = 10**bpow
            main_run(edm, initial_beta=initial_beta, target_beta=target_beta)
            if ind < n_steps - 1:
                edm.new_run()


def normal_run():
    dry_run = False
    initial_beta = 0
    target_beta = 1

    edm = ExperimentDataManager(
        experiment_name=f"fh22_{initial_beta}_target_beta",
        dry_run=dry_run,
    )

    main_run(edm, initial_beta=initial_beta, target_beta=target_beta)


if __name__ == "__main__":
    normal_run()
