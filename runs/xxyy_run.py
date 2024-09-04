from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from openfermion import (
    get_sparse_operator,
)

from chemical_models.specific_model import SpecificModel
from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
from fermionic_cooling.adiabatic_sweep import run_sweep
from fermionic_cooling.building_blocks import (
    get_cheat_couplers,
    get_Z_env,
    get_XsXbYsYb_couplers,
)
from fermionic_cooling.cooler_class import Cooler
from fermionic_cooling.utils import (
    dense_restricted_ham,
    get_closest_state,
    get_min_gap,
    ketbra,
    print_state_fidelity_to_eigenstates,
    subspace_energy_expectation,
)
from qutlet.models import FermiHubbardModel, RandomFermionicModel
from qutlet.utilities import (
    fidelity,
    jw_eigenspectrum_at_particle_number,
    jw_hartree_fock_state,
)


def __main__(edm: ExperimentDataManager):

    # model_name = "cooked/Cyclobutene_singlet_6e_12q"
    model_name = "random"
    subspace_simulation = True
    if "fh_" in model_name:
        n_electrons = [2, 2]
        model = FermiHubbardModel(
            lattice_dimensions=list(n_electrons),
            n_electrons=n_electrons,
            tunneling=1,
            coulomb=6,
        )
        n_qubits = len(model.qubits)
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
            couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
        elif "slater" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            couplers_fock_hamiltonian = start_fock_hamiltonian
    elif "cooked" in model_name:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.qubits)
        n_electrons = spm.n_electrons
        start_fock_hamiltonian = spm.current_model.quadratic_terms
        couplers_fock_hamiltonian = start_fock_hamiltonian
    elif "random" in model_name:
        n_qubits = 8
        model = RandomFermionicModel(
            n_qubits=n_qubits, neighbour_order=n_qubits, n_electrons="hf"
        )
        start_fock_hamiltonian = model.quadratic_terms
        couplers_fock_hamiltonian = start_fock_hamiltonian

    n_electrons = model.n_electrons

    start_sys_eig_energies, start_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=model.n_electrons,
        expanded=not subspace_simulation,
    )
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.qubits),
            ),
            particle_number=model.n_electrons,
            expanded=not subspace_simulation,
        )
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), model.n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    if subspace_simulation:
        sys_eig_energies, sys_eig_states = model.subspace_spectrum
    else:
        sys_eig_energies, sys_eig_states = model.spectrum

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    # initial state setting
    sys_initial_state = jw_hartree_fock_state(
        model=model, expanded=not subspace_simulation
    )
    print("BEFORE SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=not subspace_simulation,
    )
    if subspace_simulation:
        sys_initial_energy = subspace_energy_expectation(
            sys_initial_state, sys_eig_energies, sys_eig_states
        )
        sys_ground_energy_exp = subspace_energy_expectation(
            sys_ground_state, sys_eig_energies, sys_eig_states
        )
    else:
        sys_initial_energy = model.statevector_expectation(sys_initial_state)
        sys_ground_energy_exp = model.statevector_expectation(sys_ground_state)

    if subspace_simulation:
        sys_ham_matrix = dense_restricted_ham(
            model.fock_hamiltonian, n_electrons, n_sys_qubits
        )
    else:
        sys_ham_matrix = get_sparse_operator(model.fock_hamiltonian).toarray()
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
        model=model.__to_json__,
        model_name=model_name,
    )

    max_k = None

    couplers = get_XsXbYsYb_couplers(sys_qubits=sys_qubits, env_qubits=env_qubits)

    _, coupler_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=start_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )

    # couplers = get_cheat_coupler_list(
    #     sys_eig_states=couplers_sys_eig_states,
    #     env_eig_states=env_eig_states,
    #     qubits=sys_qubits + env_qubits,
    #     gs_indices=(coupler_gs_index,),
    #     noise=None,
    #     max_k=max_k,
    # )  # Interaction only on Qubit 0?

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

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

        if subspace_simulation:
            ham_start = dense_restricted_ham(
                start_fock_hamiltonian, n_electrons, n_sys_qubits
            )
        else:
            ham_start = get_sparse_operator(start_fock_hamiltonian).toarray()
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
            subspace_simulation=subspace_simulation,
        )
        sys_initial_state = final_state
    else:
        total_sweep_time = 0

    print("AFTER SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=not subspace_simulation,
    )

    cooler = Cooler(
        sys_hamiltonian=sys_ham_matrix,
        n_electrons=n_electrons,
        sys_qubits=model.qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
        subspace_simulation=subspace_simulation,
        time_evolve_method="expm",
        ancilla_split_spectrum=ancilla_split_spectrum,
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 30, "c": 40}
    weaken_coupling = 60

    start_omega = spectrum_width
    stop_omega = 0.5

    method = "bigbrain"

    edm.var_dump(
        depol_noise=depol_noise,
        subspace_simulation=subspace_simulation,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        ansatz_options=ansatz_options,
        method=method,
        spectrum_width=spectrum_width,
        start_omega=start_omega,
        stop_omega=stop_omega,
        ancilla_split_spectrum=ancilla_split_spectrum,
    )

    coupler_transitions = np.abs(
        np.array(couplers_sys_eig_energies[1:]) - couplers_sys_eig_energies[0]
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
        fidelity_threshold=0.99,
    )

    jobj = {
        "omegas": omegas,
        "fidelities": fidelities,
        "sys_energies": sys_ev_energies,
        "env_energies": env_ev_energies,
        "ansatz_options": ansatz_options,
    }
    edm.save_dict(jobj=jobj)

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
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="bigbrain_xxyy_couplers_subspace_cooling",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    use_style()
    __main__(edm)
    # model stuff
