import sys
from itertools import combinations

import cirq
import matplotlib.pyplot as plt
import numpy as np
from building_blocks import (
    get_cheat_coupler,
    get_cheat_coupler_list,
    get_cheat_sweep,
    get_perturbed_sweep,
    get_Z_env,
)
from coolerClass import Cooler
from openfermion import (
    FermionOperator,
    get_quadratic_hamiltonian,
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
    jw_spin_restrict_operator,
    spin_dicke_state,
)
from fermionic_cooling.adiabatic_sweep import fermion_to_dense, run_sweep
from fermionic_cooling.utils import (
    dense_restricted_ham,
    expectation_wrapper,
    fidelity,
    get_closest_degenerate_ground_state,
    get_min_gap,
    ketbra,
    print_state_fidelity_to_eigenstates,
    s_squared_penalty,
    state_fidelity_to_eigenstates,
    subspace_energy_expectation,
    two_tensor_partial_trace,
)


def __main__(edm: ExperimentDataManager):

    model_name = "cooked/water_singlet_6e_10q"
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
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        start_fock_hamiltonian = spm.current_model.quadratic_terms
        couplers_fock_hamiltonian = start_fock_hamiltonian

    start_sys_eig_energies, start_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
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

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )
    _, _, start_gs_index = get_closest_degenerate_ground_state(
        ref_state=sys_eig_states[:, 0],
        comp_energies=start_sys_eig_energies,
        comp_states=start_sys_eig_states,
        subspace_simulation=True,
    )
    coupler_gs_index = start_gs_index

    # initial state setting
    sys_initial_state = ketbra(start_sys_eig_states[:, start_gs_index])

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    print("BEFORE SWEEP")
    initial_pops = state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenstates=sys_eig_states,
        expanded=False,
    )
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=False,
    )
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

    edm.var_dump(
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eig_energies=sys_eig_energies,
        env_eig_energies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
        model_name=model_name,
    )

    max_k = None

    couplers = get_cheat_coupler_list(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {max_k}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    # evolution_time = 1e-3

    # call cool
    use_fast_sweep = True
    depol_noise = None
    is_noise_spin_conserving = False
    ancilla_split_spectrum = False

    if use_fast_sweep:
        sweep_time_mult = 1
        initial_ground_state = sys_initial_state
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

    print("AFTER SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=False,
    )

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
        ancilla_split_spectrum=ancilla_split_spectrum,
    )
    n_rep = 10

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    weaken_coupling = float(np.sqrt(3) / (2 * n_sys_qubits))

    # omegas = np.linspace(0.1, spectrum_width, max_k)[::-1]
    omegas = np.array([spectrum_width / 2])

    alphas = omegas / (weaken_coupling * n_sys_qubits)
    evolution_times = np.pi / alphas

    edm.var_dump(
        depol_noise=depol_noise,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        start_gs_index=start_gs_index,
        coupler_gs_index=coupler_gs_index,
        spectrum_width=spectrum_width,
        ancilla_split_spectrum=ancilla_split_spectrum,
    )
    (
        fidelities,
        sys_ev_energies,
        env_ev_energies,
        _,
    ) = cooler.zip_cool(
        n_rep=n_rep,
        alphas=alphas,
        evolution_times=evolution_times,
        sweep_values=omegas,
        fidelity_threshold=0.99,
    )

    jobj = {
        "fidelities": fidelities,
        "sys_energies": sys_ev_energies,
        "env_energies": env_ev_energies,
    }
    edm.save_dict(filename=f"cooling_strong_coupling_{model_name}", jobj=jobj)

    fig, ax = plt.subplots()

    ax.plot(range(len(list(flatten(fidelities)))), list(flatten(fidelities)))
    edm.save_figure(
        fig,
    )
    plt.show()


if __name__ == "__main__":
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="strong_coupling cooling",
        notes="looking at the obrien paper seems like I can cool with very high coupling, need to figure ts and alphas",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    use_style()
    __main__(edm)
    # model stuff
