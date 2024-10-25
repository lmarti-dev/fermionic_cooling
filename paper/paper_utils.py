from fermionic_cooling import Cooler, Thermalizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from qutlet.utilities import flatten

from qutlet.models import FermiHubbardModel
from fermionic_cooling.building_blocks import get_proj_couplers, get_Z_env
from fermionic_cooling.utils import (
    dense_subspace_hamiltonian,
    get_min_gap,
    thermal_density_matrix,
    thermal_density_matrix_at_particle_number,
    get_thermal_weights,
    pauli_mask_to_pstr,
)
from qutlet.utilities import (
    get_closest_state,
    ketbra,
    print_state_fidelity_to_eigenstates,
    subspace_size,
    jw_get_free_couplers,
)
from fermionic_cooling.adiabatic_sweep import run_sweep

from openfermion import get_sparse_operator


def get_model(model_name):
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2),
        n_electrons="half-filling",
        tunneling=1,
        coulomb=2,
    )
    if "coulomb" in model_name:
        start_model = model.coulomb_model
        couplers_model = model.non_interacting_model
    elif "slater" in model_name:
        start_model = model.non_interacting_model
        couplers_model = start_model
    return model, start_model, couplers_model


def controlled_cooling_load_plot(
    fig_filename,
    omegas,
    fidelities,
    env_energies,
    sys_eig_energies,
    tf_minus_val: int = None,
):
    if tf_minus_val is not None:
        new_env_energies = np.abs(np.array(env_energies[0]) - tf_minus_val)
        env_energies[0] = new_env_energies.astype(list)
    fig = Cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_energies,
        omegas=omegas,
        eigenspectrums=[sys_eig_energies - sys_eig_energies[0]],
    )
    return fig


def plot_fast_sweep_vs_dc(final_fids_with, final_fids_without, max_sweep_fid=None):
    fig, ax = plt.subplots()
    ms = np.arange(0, len(final_fids_with))
    ax.plot(
        ms,
        1 - final_fids_with,
        marker="x",
        label="With fast sweep",
    )
    ax.plot(
        ms,
        1 - final_fids_without,
        marker="d",
        label="Without fast sweep",
    )

    if max_sweep_fid is not None:
        ax.hlines(
            1 - max_sweep_fid,
            ms[0],
            ms[-1],
            "r",
            label="Slow sweep",
            linestyles="dashed",
        )

    ax.set_ylabel("Final infidelity")
    ax.set_xlabel(r"$d_c$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    return fig


def plot_bogo_jw_coefficients(total_coefficients, max_pauli_strs):

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("turbo", len(total_coefficients))
    for ind, coeffs in enumerate(total_coefficients):
        ax.plot(
            list(range(1, len(coeffs) + 1)),
            np.sort(np.abs(coeffs))[::-1],
            label=f"$V_{{({ind+1},0)}}: {max_pauli_strs[ind]}$",
            # marker=markers[ind % len(markers)],
            # markevery=5,
            color=cmap(ind),
        )

    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Pauli string index")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-4, 9e-2])
    ax.set_xlim([1, 3e3])
    # ax.legend(ncol=4)
    return fig


def plot_comparison_fast_sweep(
    fidelities: np.ndarray,
    env_ev_energies: np.ndarray,
    omegas: np.ndarray,
    sys_eig_energies: np.ndarray,
    is_tf_log: bool = False,
):

    fig, axes = plt.subplots(nrows=2, sharex=True)

    labels = ["with initial sweep", "without initial sweep"]

    # ind == 0 is with
    for ind, key in enumerate(("with", "without")):
        axes[0].plot(
            omegas[key],
            1 - np.array(fidelities[key]),
            label=labels[ind],
        )

        # only for thermalization
        if is_tf_log:
            axes[1].plot(
                omegas[key],
                np.abs(np.array(env_ev_energies[key]) - env_ev_energies[0, ind]),
            )
            axes[1].set_yscale("log")
        else:
            axes[1].plot(
                omegas[key],
                env_ev_energies[key],
            )
        axes[1].set_ylabel(r"$E_F/\omega$")

    all_energies = np.array(list(flatten([list(v) for v in env_ev_energies.values()])))

    axes[1].vlines(
        sys_eig_energies - sys_eig_energies[0],
        ymin=0,
        ymax=np.nanmax(all_energies[np.isfinite(all_energies)]),
        linestyle="--",
        color="r",
    )

    plt.xlim(min(omegas[key]) * 0.9, max(omegas[key]) * 1.1)

    axes[1].set_yscale("log")
    axes[1].invert_xaxis()

    axes[0].set_ylabel("Infidelity")
    axes[1].set_xlabel("$\omega$")
    axes[0].legend()

    return fig


def compare_single_run():

    subspace_simulation = True

    model, start_model, couplers_model = get_model("slater")
    sys_eig_energies, sys_eig_states = model.subspace_spectrum
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    _, couplers_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=couplers_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )

    start_gs_index = 0

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    all_couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(couplers_gs_index,),
        noise=0,
        max_k=None,
        use_pauli_x=False,
    )

    all_couplers = all_couplers[::-1]

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    sys_start_ground_state = ketbra(start_sys_eig_states[:, start_gs_index])

    sys_initial_state = sys_start_ground_state

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )
    ham_stop = sys_ham_matrix

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(sys_eig_energies, threshold=1e-6)

    depol_noise = None
    is_noise_spin_conserving = False

    initial_ground_state = sys_initial_state
    final_ground_state = sys_eig_states[:, 0]
    ham_start = dense_subspace_hamiltonian(
        start_model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    n_steps = 5
    total_sweep_time = spectrum_width / (min_gap**2)

    all_fidelities = {}
    all_env_energies = {}
    all_omegas = {}

    n_gaps = 8

    for use_fast_sweep in (True, False):

        print(f"Use fast sweep: {use_fast_sweep}")

        if use_fast_sweep:

            (
                fidelities,
                instant_fidelities,
                final_ground_state,
                final_state,
            ) = run_sweep(
                initial_state=initial_ground_state,
                ham_start=ham_start,
                ham_stop=ham_stop,
                final_ground_state=final_ground_state,
                instantaneous_ground_states=None,
                n_steps=n_steps,
                total_time=total_sweep_time * 0.1,
                get_populations=False,
                depol_noise=depol_noise,
                n_qubits=n_sys_qubits,
                is_noise_spin_conserving=is_noise_spin_conserving,
                n_electrons=model.n_electrons,
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
            sys_initial_state = sys_start_ground_state

        start_omega = 1.1 * (sys_eig_energies[n_gaps] - sys_eig_energies[0])
        stop_omega = 0.1

        max_k = n_gaps + 1

        couplers = all_couplers[:max_k]

        print(f"coupler done, max_k: {max_k}")
        print(f"number of couplers: {len(couplers)}")

        cooler = Cooler(
            sys_hamiltonian=sys_ham_matrix,
            n_electrons=model.n_electrons,
            sys_qubits=model.qubits,
            sys_ground_state=sys_ground_state,
            sys_initial_state=sys_initial_state,
            env_hamiltonian=env_ham,
            env_qubits=env_qubits,
            env_ground_state=env_ground_state,
            sys_env_coupler_data=couplers,
            verbosity=5,
            subspace_simulation=True,
        )
        print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

        ansatz_options = {"beta": 1, "mu": 20, "c": 30}
        weaken_coupling = 50

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
            n_rep=1,
            weaken_coupling=weaken_coupling,
            coupler_transitions=None,
            depol_noise=depol_noise,
            is_noise_spin_conserving=is_noise_spin_conserving,
        )
        if use_fast_sweep:
            key = "with"
        else:
            key = "without"
        all_fidelities[key] = fidelities[0]
        all_env_energies[key] = env_ev_energies[0]
        all_omegas[key] = omegas[0]

    return plot_comparison_fast_sweep(
        fidelities=all_fidelities,
        env_ev_energies=all_env_energies,
        omegas=all_omegas,
        sys_eig_energies=sys_eig_energies,
    )


def compare_sweep_subspaces(model_name: str, max_gaps: int = 6):

    subspace_simulation = True

    model, start_model, couplers_model = get_model(model_name)
    sys_eig_energies, sys_eig_states = model.subspace_spectrum
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    _, couplers_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=couplers_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )

    start_gs_index = 0

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    all_couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(couplers_gs_index,),
        noise=0,
        max_k=None,
        use_pauli_x=False,
    )

    all_couplers = all_couplers[::-1]

    final_fidelities = np.zeros((max_gaps, 2))

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    sys_start_ground_state = ketbra(start_sys_eig_states[:, start_gs_index])

    sys_initial_state = sys_start_ground_state
    print("BEFORE SWEEP")
    print_state_fidelity_to_eigenstates(
        state=sys_initial_state,
        eigenenergies=sys_eig_energies,
        eigenstates=sys_eig_states,
        expanded=not subspace_simulation,
    )

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )
    ham_stop = sys_ham_matrix

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(sys_eig_energies, threshold=1e-6)

    depol_noise = None
    is_noise_spin_conserving = False

    initial_ground_state = sys_initial_state
    final_ground_state = sys_eig_states[:, 0]
    ham_start = dense_subspace_hamiltonian(
        start_model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    n_steps = 5
    total_sweep_time = spectrum_width / (min_gap**2)

    for use_fast_sweep in (True, False):

        print(f"Use fast sweep: {use_fast_sweep}")

        if use_fast_sweep:

            (
                fidelities,
                instant_fidelities,
                final_ground_state,
                final_state,
            ) = run_sweep(
                initial_state=initial_ground_state,
                ham_start=ham_start,
                ham_stop=ham_stop,
                final_ground_state=final_ground_state,
                instantaneous_ground_states=None,
                n_steps=n_steps,
                total_time=total_sweep_time * 0.1,
                get_populations=False,
                depol_noise=depol_noise,
                n_qubits=n_sys_qubits,
                is_noise_spin_conserving=is_noise_spin_conserving,
                n_electrons=model.n_electrons,
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
            sys_initial_state = sys_start_ground_state

        for n_gaps in range(max_gaps):

            start_omega = 1.1 * (sys_eig_energies[n_gaps] - sys_eig_energies[0])
            stop_omega = 0.8

            max_k = n_gaps + 1

            couplers = all_couplers[:max_k]

            print(f"coupler done, max_k: {max_k}")
            print(f"number of couplers: {len(couplers)}")

            cooler = Cooler(
                sys_hamiltonian=sys_ham_matrix,
                n_electrons=model.n_electrons,
                sys_qubits=model.qubits,
                sys_ground_state=sys_ground_state,
                sys_initial_state=sys_initial_state,
                env_hamiltonian=env_ham,
                env_qubits=env_qubits,
                env_ground_state=env_ground_state,
                sys_env_coupler_data=couplers,
                verbosity=5,
                subspace_simulation=True,
            )
            print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

            ansatz_options = {"beta": 1, "mu": 20, "c": 30}
            weaken_coupling = 10

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
                n_rep=1,
                weaken_coupling=weaken_coupling,
                coupler_transitions=None,
                depol_noise=depol_noise,
                is_noise_spin_conserving=is_noise_spin_conserving,
            )
            final_fidelities[n_gaps, int(use_fast_sweep)] = fidelities[0][-1]

    fidelities, _, _, _ = run_sweep(
        initial_state=initial_ground_state,
        ham_start=ham_start,
        ham_stop=ham_stop,
        final_ground_state=final_ground_state,
        instantaneous_ground_states=None,
        n_steps=int(total_sweep_time * 10),
        total_time=int(total_sweep_time),
        get_populations=False,
        depol_noise=None,
        n_qubits=n_sys_qubits,
        is_noise_spin_conserving=False,
        n_electrons=model.n_electrons,
        subspace_simulation=subspace_simulation,
    )

    max_sweep_fid = fidelities[-1]

    return plot_fast_sweep_vs_dc(
        final_fids_with=final_fidelities[:, 1],
        final_fids_without=final_fidelities[:, 0],
        max_sweep_fid=max_sweep_fid,
    )


def thermalizer_run():
    target_beta = 1

    subspace_simulation = True
    model, start_model, couplers_model = get_model("slater")
    sys_eig_energies, sys_eig_states = model.subspace_spectrum
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    _, couplers_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=couplers_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )

    start_gs_index = 0

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    sys_start_ground_state = ketbra(start_sys_eig_states[:, start_gs_index])

    sys_initial_state = sys_start_ground_state

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )
    ham_stop = sys_ham_matrix

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    depol_noise = None
    is_noise_spin_conserving = False

    initial_ground_state = sys_initial_state
    final_ground_state = sys_eig_states[:, 0]
    ham_start = dense_subspace_hamiltonian(
        start_model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    use_fast_sweep = True
    depol_noise = None
    is_noise_spin_conserving = False

    # renormalize target beta with ground state
    renorm_target_beta = target_beta / np.abs(sys_eig_energies[0])
    start_target_beta = target_beta / np.abs(couplers_sys_eig_energies[0])
    env_beta = start_target_beta

    if use_fast_sweep:
        sweep_time_mult = 1

        final_ground_state = sys_eig_states[:, 0]
        ham_start = dense_subspace_hamiltonian(
            start_model.fock_hamiltonian, model.n_electrons, n_sys_qubits
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
            n_electrons=model.n_electrons,
            subspace_simulation=True,
        )
        sys_initial_state = final_state
    else:
        total_sweep_time = 0
        sys_initial_state = initial_ground_state

    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )
    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]

    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    thermal_env_density = thermal_density_matrix(
        beta=env_beta, ham=env_ham, qubits=env_qubits
    )

    thermal_sys_density = thermal_density_matrix_at_particle_number(
        beta=renorm_target_beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=model.n_electrons,
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

    max_k = 10

    couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        max_k=max_k,
        use_pauli_x=True,
    )
    n_rep = 1

    sweep_values = np.tile(np.linspace(min_gap, spectrum_width, 40), n_rep)[::-1]
    alphas = sweep_values / 2
    evolution_times = np.pi / np.abs(alphas)

    thermalizer = Thermalizer(
        beta=target_beta,
        thermal_env_density=thermal_env_density,
        thermal_sys_density=thermal_sys_density,
        sys_hamiltonian=sys_ham_matrix,
        n_electrons=model.n_electrons,
        sys_qubits=model.qubits,
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

    weights = get_thermal_weights(renorm_target_beta, sys_eig_energies, len(couplers))
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

    fig, ax = plt.subplots()

    all_fidelities = np.array(list(flatten(fidelities)))
    ax.plot(range(len(all_fidelities)), 1 - all_fidelities)
    ax.set_xlabel("Iteration step")
    ax.set_ylabel("Infidelity")

    return fig


def get_coeffs_and_maxpstr(jw_couplers):
    total_coefficients = []
    max_pauli_strs = []
    for coupler in jw_couplers:
        # coupler is paulisum
        pstr_list = list(coupler)
        coefficients = list(np.real(x.coefficient) for x in pstr_list)
        total_coefficients.append(np.array(coefficients))

        max_ind = np.argmax(coefficients)
        max_pstr = pstr_list[max_ind]
        pstr_pretty = pauli_mask_to_pstr(
            max_pstr.dense(qubits=max_pstr.qubits).pauli_mask, max_pstr.qubits
        )
        max_pauli_strs.append(pstr_pretty)
    return total_coefficients, max_pauli_strs


def free_couplers_paulis(max_k:int):
    model, _, _ = get_model("slater")
    jw_couplers = jw_get_free_couplers(model=model, max_k=max_k)
    total_coefficients, max_pauli_strs = get_coeffs_and_maxpstr(jw_couplers)
    return plot_bogo_jw_coefficients(
        total_coefficients=total_coefficients, max_pauli_strs=max_pauli_strs
    )
