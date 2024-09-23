from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from openfermion import FermionOperator, get_sparse_operator

from chemical_models.specific_model import SpecificModel
from data_manager import ExperimentDataManager
from fauplotstyle.styler import style
from fermionic_cooling.adiabatic_sweep import run_sweep
from fermionic_cooling.building_blocks import (
    get_cheat_couplers,
    get_GivensX_couplers,
    get_Z_env,
)

from fermionic_cooling.cooling_ansatz import CoolingAnsatz, fridge_energy_objective
from fermionic_cooling.cooler_class import Cooler
from fermionic_cooling.utils import (
    dense_restricted_ham,
    get_closest_state,
    get_min_gap,
    ketbra,
    print_state_fidelity_to_eigenstates,
    subspace_energy_expectation,
)
from qutlet.models import FermiHubbardModel, FermionicModel
from qutlet.utilities import (
    fidelity,
    jw_eigenspectrum_at_particle_number,
    jw_hartree_fock_state,
    subspace_size,
    gaussian_envelope,
)

from qutlet.optimisers import ScipyOptimisers
from qutlet.objectives.statevector_objectives import energy_objective


def __main__(edm: ExperimentDataManager):

    # model_name = "cooked/sulfanium_triplet_6e_12q"
    model_name = "fh_slater"
    initial_state_name = "hartreefock"
    subspace_simulation = True
    model_name, model, start_fock_hamiltonian, couplers_fock_hamiltonian = pick_model(
        model_name
    )

    n_electrons = model.n_electrons

    couplers_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_sys_qubits, model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_eig_energies, sys_eig_states = model.subspace_spectrum

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    start_gs_index, sys_initial_state = get_initial_state(
        initial_state_name,
        subspace_simulation,
        model,
        start_fock_hamiltonian,
        sys_ground_state,
    )

    coupler_gs_index = start_gs_index

    print("BEFORE SWEEP")
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
        model=model.__to_json__,
        model_name=model_name,
        subspace_simulation=subspace_simulation,
        initial_state_name=initial_state_name,
    )

    max_k = None

    couplers = get_GivensX_couplers(sys_qubits, env_qubits)
    couplers = get_cheat_couplers(
        sys_eig_states=sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
        use_pauli_x=False,
    )

    couplers = [
        sum(
            [x * w for x, w in zip(couplers, gaussian_envelope(mu, 0.1, len(couplers)))]
        )
        for mu in np.linspace(0.1, 0.9, 100)
    ]

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    # evolution_time = 1e-3

    # call cool
    use_fast_sweep = False
    depol_noise = None
    is_noise_spin_conserving = False

    sys_initial_state = run_fast_sweep(
        subspace_simulation,
        start_fock_hamiltonian,
        n_electrons,
        n_sys_qubits,
        sys_eig_energies,
        sys_eig_states,
        sys_initial_state,
        sys_ham_matrix,
        spectrum_width,
        use_fast_sweep,
        depol_noise,
        is_noise_spin_conserving,
    )

    cooler = CoolingAnsatz(
        model=model,
        sys_hamiltonian=sys_ham_matrix,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=-1,
        subspace_simulation=subspace_simulation,
        time_evolve_method="expm",
    )
    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    edm.var_dump(
        depol_noise=depol_noise,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        start_gs_index=start_gs_index,
        coupler_gs_index=coupler_gs_index,
        spectrum_width=spectrum_width,
        min_gap=min_gap,
    )
    adapt_steps = 10

    objective = fridge_energy_objective(cooler=cooler)
    objective = energy_objective(model=model)
    optimiser = ScipyOptimisers(ansatz=cooler, objective=objective)

    env_cooled_energies = np.zeros((adapt_steps, len(couplers)))
    for step in range(adapt_steps):
        for ind in range(len(couplers)):

            env_coupling = spectrum_width / 2
            alpha = env_coupling / (cooler.weaken_coupling * n_sys_qubits)
            evolution_time = np.pi / alpha
            cooler.sys_env_coupler_easy_setter(ind, None)
            if len(cooler.picked_couplers):
                total_density_matrix = cooler.simulate()
            else:
                total_density_matrix = cooler.total_initial_state
            (
                sys_cooled_fidelity,
                sys_cooled_energy,
                env_cooled_energy,
                _,
            ) = cooler.cooling_step(
                total_density_matrix=total_density_matrix,
                alpha=alpha,
                env_coupling=env_coupling,
                evolution_time=evolution_time,
            )
            print(
                f"coupler {ind} fid: {sys_cooled_fidelity:.5f}, esys: {sys_cooled_energy:.5f}, eenv: {env_cooled_energy:.5f}",
                end="\r",
            )
            env_cooled_energies[step, ind] = env_cooled_energy
        max_ind = np.argmax(env_cooled_energies[step, :])
        print(
            f"coupler {ind} fid: {sys_cooled_fidelity:.5f}, esys: {sys_cooled_energy:.5f}, eenv: {env_cooled_energy:.5f}"
        )
        cooler.picked_couplers.append(couplers[max_ind])
        cooler.params.append(env_coupling)
        cooler.symbols.append(None)
        result, sim_data = optimiser.optimise(
            initial_params="random",
            initial_state=cooler.sys_initial_state,
            bounds=[(min_gap, spectrum_width) for _ in range(len(cooler.symbols))],
        )
        cooler.params = list(result["x"])
        out_state = cooler.simulate()
        final_fid = cooler.sys_fidelity(
            cooler.partial_trace_wrapper(out_state, trace_out="env")
        )
        print(f"max coupler {ind} optimized fid {final_fid:.5f}")


def get_initial_state(
    initial_state_name: str,
    subspace_simulation: bool,
    model: FermionicModel,
    start_fock_hamiltonian: FermionOperator,
    sys_ground_state: np.ndarray,
):
    if initial_state_name == "slater":
        start_sys_eig_energies, start_sys_eig_states = (
            jw_eigenspectrum_at_particle_number(
                sparse_operator=get_sparse_operator(
                    start_fock_hamiltonian,
                    n_qubits=len(model.qubits),
                ),
                particle_number=model.n_electrons,
                expanded=False,
            )
        )

        _, start_gs_index = get_closest_state(
            ref_state=sys_ground_state,
            comp_states=start_sys_eig_states,
            subspace_simulation=subspace_simulation,
        )
        sys_initial_state = ketbra(start_sys_eig_states[:, start_gs_index])
    elif initial_state_name == "hartreefock":
        sys_initial_state = jw_hartree_fock_state(
            model=model, expanded=not subspace_simulation
        )
        start_gs_index = 0

    return start_gs_index, sys_initial_state


def run_fast_sweep(
    subspace_simulation,
    start_fock_hamiltonian,
    n_electrons,
    n_sys_qubits,
    sys_eig_energies,
    sys_eig_states,
    sys_initial_state,
    sys_ham_matrix,
    spectrum_width,
    use_fast_sweep,
    depol_noise,
    is_noise_spin_conserving,
):
    if use_fast_sweep:
        sweep_time_mult = 1
        initial_ground_state = sys_initial_state
        final_ground_state = sys_eig_states[:, 0]
        ham_start = dense_restricted_ham(
            start_fock_hamiltonian, n_electrons, n_sys_qubits
        )
        ham_stop = sys_ham_matrix
        n_steps = 30
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
        print("AFTER SWEEP")
        print_state_fidelity_to_eigenstates(
            state=sys_initial_state,
            eigenenergies=sys_eig_energies,
            eigenstates=sys_eig_states,
            expanded=False,
        )
    else:
        total_sweep_time = 0
    return sys_initial_state


def pick_model(model_name):
    if "fh_" in model_name:
        model = FermiHubbardModel(
            lattice_dimensions=(2, 2),
            n_electrons="half-filling",
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
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.qubits)
        n_electrons = spm.n_electrons
        start_fock_hamiltonian = spm.current_model.quadratic_terms
        couplers_fock_hamiltonian = start_fock_hamiltonian
    return model_name, model, start_fock_hamiltonian, couplers_fock_hamiltonian


if __name__ == "__main__":
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="adaptive_cooling",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    style()
    __main__(edm)
    # model stuff
