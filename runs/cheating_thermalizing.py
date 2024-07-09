import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from building_blocks import (
    get_cheat_coupler_list,
    get_cheat_sweep,
    get_cheat_thermalizers,
    get_matrix_coupler,
    get_Z_env,
)
from openfermion import get_sparse_operator, jw_hartree_fock_state
from thermalizer import Thermalizer

from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, qmap, spin_dicke_state
from fermionic_cooling.utils import (
    dense_restricted_ham,
    get_min_gap,
    ketbra,
    state_fidelity_to_eigenstates,
    thermal_density_matrix,
    thermal_density_matrix_at_particle_number,
    spin_dicke_mixed_state,
    print_state_fidelity_to_eigenstates,
)

from fermionic_cooling.plotting.plot_loopcool_thermalizing_state import (
    plot_fid_progression,
)

from adiabatic_sweep import run_sweep


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
    fids_initl = state_fidelity_to_eigenstates(
        sys_initial_state, sys_eig_states, expanded=False
    )
    fids_final = state_fidelity_to_eigenstates(
        final_sys_density_matrix, sys_eig_states, expanded=False
    )
    fids_thermal = state_fidelity_to_eigenstates(
        thermal_sys_density, sys_eig_states, expanded=False
    )

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


def get_thermal_weights(beta: float, sys_eig_energies: np.ndarray, max_k=None):
    if max_k is None:
        weights = [np.exp(-beta * x) for x in sys_eig_energies[1:]]
    elif isinstance(max_k, list):
        weights = [np.exp(-beta * x) for x in sys_eig_energies[max_k]]
    elif isinstance(max_k, int):
        weights = [np.exp(-beta * x) for x in sys_eig_energies[1 : max_k + 1]]
    return np.array(weights) / np.sum(weights)


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


def main_run(edm: ExperimentDataManager, initial_beta, target_beta, **kwargs):
    # whether we want to skip all saving data

    # model stuff

    gs_index = 2
    model_name = "fh_coulomb"
    if "fh_" in model_name:
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
            couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
        elif "slater" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            couplers_fock_hamiltonian = start_fock_hamiltonian
    # define inverse temp

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    n_env_qubits = 1
    n_total_qubits = n_sys_qubits + n_env_qubits

    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    sys_ham_matrix = dense_restricted_ham(
        model.fock_hamiltonian, n_electrons, n_sys_qubits
    )

    couplers_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )
    )
    start_sys_eig_energies, start_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    spectrum_width = np.abs(np.max(sys_eig_energies) - np.min(sys_eig_energies))
    min_gap = get_min_gap(sys_eig_energies, threshold=1e-8)
    # ketbra(sys_slater_state)
    spin_dicke_initial_state = spin_dicke_mixed_state(
        n_qubits=n_sys_qubits, Nf=n_electrons
    )

    use_fast_sweep = True
    depol_noise = None
    is_noise_spin_conserving = False

    # renormalize target beta with ground state
    renorm_target_beta = target_beta / np.abs(sys_eig_energies[0])
    start_target_beta = target_beta / np.abs(couplers_sys_eig_energies[0])
    env_beta = start_target_beta

    initial_ground_state = start_sys_eig_states[:, gs_index]

    # initial_ground_state = free_sys_eig_states[:, gs_index]

    if use_fast_sweep:
        sweep_time_mult = 1

        final_ground_state = sys_eig_states[:, 0]
        ham_start = dense_restricted_ham(
            start_fock_hamiltonian, n_electrons, n_sys_qubits
        )
        ham_stop = sys_ham_matrix
        n_steps = 10
        total_sweep_time = (
            sweep_time_mult
            * spectrum_width
            / (get_min_gap(sys_eig_energies, threshold=1e-12) ** 2)
        )

        (
            fidelities,
            instant_fidelities,
            final_ground_state,
            populations,
            final_state,
        ) = run_sweep(
            initial_state=initial_ground_state,
            ham_start=ham_start,
            ham_stop=ham_stop,
            final_ground_state=final_ground_state,
            instantaneous_ground_states=None,
            n_steps=n_steps,
            total_time=total_sweep_time,
            get_populations=True,
            depol_noise=depol_noise,
            n_qubits=n_sys_qubits,
            is_noise_spin_conserving=is_noise_spin_conserving,
            n_electrons=n_electrons,
            subspace_simulation=True,
        )
        sys_initial_state = final_state
    else:
        total_sweep_time = 0
        sys_initial_state = initial_ground_state

    print("AFTER SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=False,
    )

    couplers_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )
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
        beta=renorm_target_beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=False,
    )
    # Important part of the script where things that matter happen

    print("Thermal state dist")

    print_state_fidelity_to_eigenstates(
        thermal_sys_density,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=False,
    )

    thermal_sys_initial_density = thermal_density_matrix_at_particle_number(
        beta=initial_beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=False,
    )

    # sys_initial_state = thermal_sys_initial_density
    max_k = list(range(1, 25))
    # max_k = 16
    couplers = get_cheat_coupler_list(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        max_k=max_k,
        use_pauli_x=True,
    )
    weights = get_thermal_weights(renorm_target_beta, sys_eig_energies, max_k)
    weights = None

    n_rep = 10

    gaps = np.array(np.array(sys_eig_energies) - sys_eig_energies[0])
    indices = [1, 2, 8, 13]
    sweep_values = np.tile(gaps[indices], n_rep)[::-1]
    # sweep_values = get_cheat_sweep(spectrum=sys_eig_energies)
    alphas = sweep_values
    evolution_times = np.pi / np.abs(alphas)

    fridge_thermal_energy = env_ham.expectation_from_density_matrix(
        thermal_env_density, qubit_map={k: v for v, k in enumerate(env_qubits)}
    )

    thermalizer = Thermalizer(
        beta=target_beta,
        thermal_env_density=thermal_env_density,
        thermal_sys_density=thermal_sys_density,
        sys_hamiltonian=sys_ham_matrix,
        n_electrons=n_electrons,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
        subspace_simulation=True,
        time_evolve_method="expm",
    )
    edm.var_dump(
        initial_beta=initial_beta,
        target_beta=target_beta,
        env_beta=env_beta,
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eigenspectrum=sys_eig_energies,
        env_eigenergies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
        use_fast_sweep=use_fast_sweep,
        depol_noise=depol_noise,
        is_noise_spin_conserving=is_noise_spin_conserving,
    )

    # probe_times(edm, cooler, alphas, sweep_values)
    method = "zipcool"
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
            use_random_coupler=True,
            weights=weights,
        )

        jobj = {
            "alphas": alphas,
            "omegas": sweep_values,
            "evolution_times": evolution_times,
            "fidelities": fidelities,
            "sys_energies": sys_energies,
            "env_energies": env_energies,
            # "final_sys_density_matrix": final_sys_density_matrix,
        }
        edm.save_dict(
            jobj=jobj,
            filename=f"thermalizin_free_{model_name}_{gs_index}",
            add_timestamp=False,
        )

        print_thermalizing_stats(
            renorm_target_beta,
            sys_initial_state,
            sys_energies,
            sys_eig_states,
            sys_eig_energies,
            final_sys_density_matrix,
            thermal_sys_density,
            env_energies,
            fidelities,
        )
        plot_cool = "fid"
        if plot_cool == "default":
            fig = thermalizer.plot_default_cooling(
                omegas=sweep_values,
                fidelities=fidelities[1:],
                env_energies=env_energies[1:],
                suptitle=f"Thermalizing {model.x_dimension}$\\times${model.y_dimension} Fermi-Hubbard",
                n_rep=n_rep,
            )
        elif plot_cool == "fid":
            fig = plot_fid_progression(fids=fidelities)
        edm.save_figure(
            fig,
            filename=f"thermalizing_free_{model_name}_{gs_index}",
            add_timestamp=False,
        )
        plt.show()

    elif method == "bigbrain":
        ansatz_options = {
            "beta": 1,
            "mu": 40,
            "c": 20,
            "minus": fridge_thermal_energy,
            "clamp": False,
        }
        weaken_coupling = 30

        start_omega = 4

        stop_omega = 0.3

        n_rep = 1
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
            use_random_coupler=False,
        )

        jobj = {
            "omegas": omegas,
            "fidelities": fidelities,
            "sys_energies": sys_ev_energies,
            "env_energies": env_ev_energies,
            "ansatz_options": ansatz_options,
        }
        edm.save_dict(filename=f"cooling_free_{gs_index}", jobj=jobj)

        dashes = "diff"
        if dashes == "diff":
            spectrums = np.diff(sys_eig_energies)
        elif dashes == "gaps":
            spectrums = np.array(sys_eig_energies) - sys_eig_energies[0]

        fig = thermalizer.plot_controlled_cooling(
            fidelities=fidelities,
            env_energies=env_ev_energies,
            omegas=omegas,
            eigenspectrums=[
                spectrums,
            ],
            substract_energy=fridge_thermal_energy,
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
    use_style()
    normal_run()
