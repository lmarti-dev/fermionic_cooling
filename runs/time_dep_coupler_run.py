from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from openfermion import (
    get_sparse_operator,
)

from chemical_models.specific_model import SpecificModel
from data_manager import ExperimentDataManager
from fauplotstyle.styler import style
from fermionic_cooling.adiabatic_sweep import run_sweep
from fermionic_cooling.building_blocks import (
    get_cheat_couplers,
    get_Z_env,
    get_XsXbYsYb_coupler,
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
from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from qutlet.utilities import (
    fidelity,
    jw_eigenspectrum_at_particle_number,
    jw_hartree_fock_state,
    gaussian_envelope,
)


from fermionic_cooling.filter_functions import (
    get_ding_filter_function,
    get_lloyd_filter_function,
)


def get_test_ff(a):
    def filter_function(t):
        return np.exp(-t * a)

    return filter_function


def __main__(edm: ExperimentDataManager):

    # model_name = "cooked/Fe3_NTA_doublet_3e_8q"
    model_name = "fh_slater"
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

    n_electrons = model.n_electrons
    start_sys_eig_energies, start_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )
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
    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")
    subspace_simulation = True

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=not subspace_simulation,
    )

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    _, start_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=start_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )
    coupler_gs_index = start_gs_index

    initial_state_name = "hartreefock"

    if initial_state_name == "slater":
        sys_initial_state = ketbra(start_sys_eig_states[:, start_gs_index])
    elif initial_state_name == "hartreefock":
        sys_initial_state = jw_hartree_fock_state(
            model=model, expanded=not subspace_simulation
        )

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

    couplers = get_cheat_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
        use_pauli_x=False,
    )
    weights = gaussian_envelope(mu=0.0, sigma=0.5, n_steps=len(couplers))
    couplers = [sum([x * w for x, w in zip(couplers, weights)])]
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
    ancilla_split_spectrum = False

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
            n_qubits=n_sys_qubits,
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
    total_sim_time = 2.58
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
    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    edm.var_dump(
        depol_noise=depol_noise,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        method="time dep coupling",
        start_gs_index=start_gs_index,
        coupler_gs_index=coupler_gs_index,
        spectrum_width=spectrum_width,
        min_gap=min_gap,
        ancilla_split_spectrum=ancilla_split_spectrum,
    )

    times = np.linspace(0.01, total_sim_time, 31)

    # filter_function = get_test_ff(a=0.1)
    # filter_function = get_lloyd_filter_function(
    #     biga=1, beta=total_sim_time / 3, tau=total_sim_time / 2
    # )
    total_plot_times = []
    total_fidelities = []
    total_sys_energies = []

    total_env_energies = []
    n_reps = 100
    last_plot_times = 0

    edm.var_dump(
        total_sim_time=total_sim_time,
        n_reps=n_reps,
    )

    for rep in range(n_reps):
        fac = 1
        filter_function = get_ding_filter_function(
            a=2.5 * spectrum_width * fac,
            da=0.5 * spectrum_width * fac,
            b=min_gap,
            db=min_gap,
        )
        (
            plot_times,
            fidelities,
            sys_ev_energies,
            env_ev_energies,
            total_density_matrix,
        ) = cooler.time_cool(
            filter_function=filter_function,
            times=times,
            env_coupling=spectrum_width,
            alpha=1,
        )

        plot_times_repped = plot_times + last_plot_times

        total_plot_times.extend(plot_times_repped)
        total_fidelities.extend(fidelities)
        total_env_energies.extend(env_ev_energies)
        total_sys_energies.extend(sys_ev_energies)

        last_plot_times = plot_times_repped[-1]

        cooler.sys_initial_state = cooler.partial_trace_wrapper(
            total_density_matrix, trace_out="env"
        )
        if fidelities[-1] > 0.99:
            break

    jobj = {
        "total_plot_times": total_plot_times,
        "total_fidelities": total_fidelities,
        "total_sys_energies": total_sys_energies,
        "total_env_energies": total_env_energies,
        "alphas": [filter_function(t) for t in times],
    }
    edm.save_dict(jobj=jobj)

    edm.var_dump(filter_function=filter_function.__name__)

    fig = cooler.plot_time_cooling(
        np.linspace(0, n_reps * total_sim_time * len(couplers), len(total_fidelities)),
        total_fidelities,
        total_env_energies,
    )
    edm.save_figure(
        fig,
    )
    plt.show()


if __name__ == "__main__":
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="time_dependent_coupler_cooling",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    style()
    __main__(edm)
    # model stuff
