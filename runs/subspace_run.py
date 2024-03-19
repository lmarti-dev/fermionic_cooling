import sys

import cirq
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
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
    FermionOperator,
    get_sparse_operator,
    jw_hartree_fock_state,
)

from chemical_models.specificModel import SpecificModel
from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import (
    flatten,
    jw_eigenspectrum_at_particle_number,
    spin_dicke_state,
    jw_spin_restrict_operator,
)
from fermionic_cooling.adiabatic_sweep import fermion_to_dense, run_sweep
from fermionic_cooling.utils import (
    expectation_wrapper,
    get_min_gap,
    ketbra,
    s_squared_penalty,
    state_fidelity_to_eigenstates,
    subspace_energy_expectation,
    fidelity,
    two_tensor_partial_trace,
    dense_restricted_ham,
)


def __main__(args):
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="fh_bigbrain_subspace",
        notes="fh cooling with subspace simulation, aiming for larger systems",
        dry_run=dry_run,
    )
    use_style()
    # model stuff

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

    gs_index = 2
    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            couplers_fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))
    sys_hartree_fock = np.zeros((subspace_dim,))
    sys_hartree_fock[0] = 1

    sys_slater_state = free_sys_eig_states[:, gs_index]

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    # initial state setting
    sys_initial_state = ketbra(sys_slater_state)

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    eig_fids = state_fidelity_to_eigenstates(
        state=sys_initial_state, eigenstates=sys_eig_states, expanded=False
    )
    print("Initial populations")
    for fid, sys_eig_energy in zip(eig_fids, sys_eig_energies):
        print(
            f"fid: {np.abs(fid):.4f} gap: {np.abs(sys_eig_energy-sys_eig_energies[0]):.3f}"
        )
    print(f"sum fids {sum(eig_fids)}")
    sys_initial_energy = subspace_energy_expectation(
        sys_initial_state, sys_eig_energies, sys_eig_states
    )
    sys_ground_energy_exp = subspace_energy_expectation(
        sys_ground_state, sys_eig_energies, sys_eig_states
    )

    sys_ham_matrix = dense_restricted_ham(
        model.fock_hamiltonian, n_electrons, n_sys_qubits
    )

    initial_fid = fidelity(
        sys_initial_state,
        sys_ground_state,
    )
    print("initial fidelity: {}".format(initial_fid))
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

    max_k = 6

    couplers = get_cheat_coupler_list(
        sys_eig_states=free_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(gs_index,),
        noise=0,
        max_k=max_k,
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(free_sys_eig_energies, threshold=1e-6)

    # evolution_time = 1e-3

    # call cool
    use_fast_sweep = True
    depol_noise = None
    is_noise_spin_conserving = False

    if use_fast_sweep:
        sweep_time_mult = 0.01
        initial_ground_state = ketbra(sys_slater_state)
        final_ground_state = sys_eig_states[:, 0]
        ham_start = dense_restricted_ham(
            start_fock_hamiltonian, n_electrons, n_sys_qubits
        )
        ham_stop = sys_ham_matrix
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
            n_qubits=n_sys_qubits,
            is_noise_spin_conserving=is_noise_spin_conserving,
            n_electrons=n_electrons,
            subspace_simulation=True,
        )
        sys_initial_state = final_state
    else:
        total_sweep_time = 0

    cooler = Cooler(
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
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 30, "c": 20}
    weaken_coupling = 100

    start_omega = 1.75
    stop_omega = 0.3

    method = "bigbrain"

    edm.dump_some_variables(
        depol_noise=depol_noise,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        ansatz_options=ansatz_options,
        method=method,
    )

    coupler_transitions = np.abs(
        np.array(free_sys_eig_energies[1:]) - free_sys_eig_energies[0]
    )

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
        is_noise_spin_conserving=is_noise_spin_conserving,
        use_random_coupler=False,
    )

    jobj = {
        "omegas": omegas,
        "fidelities": fidelities,
        "sys_energies": sys_ev_energies,
        "env_energies": env_ev_energies,
        "ansatz_options": ansatz_options,
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


if __name__ == "__main__":
    __main__(sys.argv)
