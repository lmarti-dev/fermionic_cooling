import sys

import cirq
import matplotlib.pyplot as plt
import numpy as np
from adiabatic_sweep import fermion_to_dense, run_sweep
from building_blocks import (
    get_cheat_coupler,
    get_cheat_coupler_list,
    get_cheat_sweep,
    get_perturbed_sweep,
    get_Z_env,
)
from coolerClass import Cooler
from openfermion import (
    get_quadratic_hamiltonian,
    get_sparse_operator,
)

from fermionic_cooling.utils import (
    expectation_wrapper,
    get_min_gap,
    ketbra,
    print_state_fidelity_to_eigenstates,
)

from chemical_models.specificModel import SpecificModel
from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
import multiprocessing as mp


def comparison_ngaps(edm, n_gaps):
    # whether we want to skip all saving data

    # model stuff

    model_name = "fh_coulomb"
    if "fh_" in model_name:
        model = FermiHubbardModel(x_dimension=3, y_dimension=3, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [3, 3]
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
            couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
        elif "slater" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            couplers_fock_hamiltonian = start_fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        start_fock_hamiltonian = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    couplers_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=True,
        )
    )

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    _, start_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    start_gs_index = 0
    sys_start_state = start_eig_states[:, start_gs_index]
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    start_omega = 1.1 * (sys_eig_energies[n_gaps] - sys_eig_energies[0])
    stop_omega = 0.8
    # initial state setting
    sys_initial_state = ketbra(sys_start_state)

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    print("BEFORE SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=False,
    )
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

    edm.var_dump(
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eig_energies=sys_eig_energies,
        env_eig_energies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )
    max_k = n_gaps + 1
    couplers = get_cheat_coupler_list(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(start_gs_index,),
        noise=0,
        max_k=max_k,
        use_pauli_x=False,
    )  # Interaction only on Qubit 0?
    print(f"coupler done, max_k: {max_k}")
    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values

    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    # evolution_time = 1e-3

    fpaths = []
    sweep_time_mult = 0.01
    depol_noise = 1e-4
    is_noise_spin_conserving = False

    for which_initial_process in ("adiabatic", "none"):
        print(f"Initial process: {which_initial_process}")

        if which_initial_process == "adiabatic":
            # call sweep
            initial_ground_state = ketbra(sys_start_state)
            final_ground_state = sys_eig_states[:, 0]
            ham_start = fermion_to_dense(start_fock_hamiltonian)
            ham_stop = fermion_to_dense(model.fock_hamiltonian)
            n_steps = 5
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
                is_noise_spin_conserving=is_noise_spin_conserving,
            )
            sys_initial_state = final_state
        elif which_initial_process == "none":
            sys_initial_state = sys_start_state
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

        print("after sweep")
        print_state_fidelity_to_eigenstates(
            state=sys_initial_state,
            eigenenergies=sys_eig_energies,
            eigenstates=sys_eig_states,
            expanded=False,
        )

        n_rep = 1

        print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

        ansatz_options = {"beta": 1, "mu": 30, "c": 40}
        weaken_coupling = 10

        dump_vars = {
            "start_omega": start_omega,
            "stop_omega": stop_omega,
            "sweep_time_mult": sweep_time_mult,
            "n_gaps": n_gaps,
            "ansatz_options": ansatz_options,
            "total_sweep_time": total_sweep_time,
            "total_cool_time": cooler.total_cooling_time,
            "weaken_coupling": weaken_coupling,
        }
        edm.var_dump(**dump_vars)

        (
            fidelities,
            sys_ev_energies,
            omegas,
            env_ev_energies,
            _,
        ) = cooler.big_brain_cool(
            start_omega=start_omega,
            stop_omega=stop_omega,
            ansatz_options=ansatz_options,
            n_rep=n_rep,
            weaken_coupling=weaken_coupling,
            coupler_transitions=None,
            depol_noise=depol_noise,
            is_noise_spin_conserving=is_noise_spin_conserving,
        )

        jobj = {
            "omegas": omegas,
            "fidelities": fidelities,
            "sys_energies": sys_ev_energies,
            "env_ev_energies": env_ev_energies,
        }
        edm.save_dict(
            filename=f"cooling_free_{which_initial_process}_n_gaps_{n_gaps}",
            jobj=jobj,
            return_fpath=True,
        )

    # plot_results(edm, fpaths[0], fpaths[1], sys_eig_energies)


if __name__ == "__main__":
    use_style()

    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="cooling_with_initial_adiab_sweep",
        notes="adding an initial sweep before the cooling run",
        dry_run=dry_run,
    )
    # pool = mp.Pool(mp.cpu_count())
    # n_gaps = list(range(1, 36))
    # result = pool.starmap(comparison_ngaps, zip([edm] * len(n_gaps), n_gaps))

    comparison_ngaps(edm, 6)
