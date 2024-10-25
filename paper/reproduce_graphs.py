import numpy as np

from fermionic_cooling import Cooler
from fermionic_cooling.building_blocks import get_proj_couplers, get_Z_env
from fermionic_cooling.utils import (
    dense_subspace_hamiltonian,
    get_min_gap,
    subspace_energy_expectation,
)
from qutlet.utilities import (
    get_closest_state,
    jw_hartree_fock_state,
    ketbra,
    print_state_fidelity_to_eigenstates,
    subspace_size,
    fidelity_wrapper,
)
from fermionic_cooling.adiabatic_sweep import run_sweep
import matplotlib.pyplot as plt

from fermionic_cooling.paper.paper_utils import (
    get_model,
    compare_sweep_subspaces,
    thermalizer_run,
    compare_single_run,
    free_couplers_paulis,
)


def figure_3(save: bool = True):

    subspace_simulation = True

    model, start_model, couplers_model = get_model("slater")
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_eig_energies, sys_eig_states = model.subspace_spectrum

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

    sys_initial_energy = subspace_energy_expectation(
        sys_initial_state, sys_eig_energies, sys_eig_states
    )
    sys_ground_energy_exp = subspace_energy_expectation(
        sys_ground_state, sys_eig_energies, sys_eig_states
    )

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    initial_fid = fidelity_wrapper(
        sys_initial_state, sys_ground_state, subspace_simulation=True
    )
    print("initial fidelity: {}".format(initial_fid))
    print("ground energy from spectrum: {}".format(sys_ground_energy))
    print("ground energy from model: {}".format(sys_ground_energy_exp))
    print("initial energy from model: {}".format(sys_initial_energy))

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    max_k = None

    couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
        use_pauli_x=False,
    )

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    # call cool
    depol_noise = None
    is_noise_spin_conserving = False

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
        subspace_simulation=subspace_simulation,
        time_evolve_method="expm",
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 20, "c": 30}
    weaken_coupling = 60

    start_omega = spectrum_width * 1.1
    stop_omega = min_gap * 0.5

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

    fig = cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_ev_energies,
        omegas=omegas,
        eigenspectrums=[
            sys_eig_energies - sys_eig_energies[0],
        ],
    )
    if save:
        fig.savefig("./fig3.png")
    plt.show()


def figure_7(save: bool = True):

    subspace_simulation = True

    model, start_model, couplers_model = get_model("slater")
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_eig_energies, sys_eig_states = model.subspace_spectrum

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    _, start_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=start_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )
    coupler_gs_index = start_gs_index

    initial_state_name = "slater"

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
        expanded=not subspace_simulation,
    )
    sys_initial_energy = subspace_energy_expectation(
        sys_initial_state, sys_eig_energies, sys_eig_states
    )
    sys_ground_energy_exp = subspace_energy_expectation(
        sys_ground_state, sys_eig_energies, sys_eig_states
    )

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    initial_fid = fidelity_wrapper(
        sys_initial_state, sys_ground_state, subspace_simulation=True
    )
    print("initial fidelity: {}".format(initial_fid))
    print("ground energy from spectrum: {}".format(sys_ground_energy))
    print("ground energy from model: {}".format(sys_ground_energy_exp))
    print("initial energy from model: {}".format(sys_initial_energy))

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    max_k = 32

    couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
        use_pauli_x=False,
    )

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    # call cool
    use_fast_sweep = True
    depol_noise = 1e-4
    is_noise_spin_conserving = False

    if use_fast_sweep:
        sweep_time_mult = 1
        initial_ground_state = sys_initial_state
        final_ground_state = sys_eig_states[:, 0]
        ham_start = dense_subspace_hamiltonian(
            start_model.fock_hamiltonian, model.n_electrons, n_sys_qubits
        )
        ham_stop = sys_ham_matrix
        n_steps = 4
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
        subspace_simulation=subspace_simulation,
        time_evolve_method="expm",
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 30, "c": 40}
    weaken_coupling = 50

    start_omega = 1.75
    stop_omega = 0.4

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

    fig = cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_ev_energies,
        omegas=omegas,
        eigenspectrums=[
            sys_eig_energies - sys_eig_energies[0],
        ],
    )
    if save:
        fig.savefig("./fig7.png")
    plt.show()


def figure_8(save: bool = True):

    subspace_simulation = True

    model, start_model, couplers_model = get_model("slater")
    start_sys_eig_energies, start_sys_eig_states = start_model.subspace_spectrum
    couplers_sys_eig_energies, couplers_sys_eig_states = (
        couplers_model.subspace_spectrum
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)
    subspace_dim = subspace_size(n_qubits=model.n_qubits, n_electrons=model.n_electrons)
    print(f"SUBSPACE: {subspace_dim} matrices will be {subspace_dim**2}")

    sys_eig_energies, sys_eig_states = model.subspace_spectrum

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
        expanded=not subspace_simulation,
    )
    sys_initial_energy = subspace_energy_expectation(
        sys_initial_state, sys_eig_energies, sys_eig_states
    )
    sys_ground_energy_exp = subspace_energy_expectation(
        sys_ground_state, sys_eig_energies, sys_eig_states
    )

    sys_ham_matrix = dense_subspace_hamiltonian(
        model.fock_hamiltonian, model.n_electrons, n_sys_qubits
    )

    initial_fid = fidelity_wrapper(
        sys_initial_state, sys_ground_state, subspace_simulation=True
    )
    print("initial fidelity: {}".format(initial_fid))
    print("ground energy from spectrum: {}".format(sys_ground_energy))
    print("ground energy from model: {}".format(sys_ground_energy_exp))
    print("initial energy from model: {}".format(sys_initial_energy))

    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    max_k = None

    couplers = get_proj_couplers(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
        use_pauli_x=False,
    )

    print("coupler done")

    print(f"number of couplers: {len(couplers)}")

    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)
    min_gap = get_min_gap(couplers_sys_eig_energies, threshold=1e-6)

    # call cool
    depol_noise = 1e-5
    is_noise_spin_conserving = True

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
        subspace_simulation=subspace_simulation,
        time_evolve_method="expm",
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 30, "c": 20}
    weaken_coupling = 100

    start_omega = spectrum_width * 1.1
    stop_omega = min_gap * 0.5

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

    fig = cooler.plot_controlled_cooling(
        fidelities=fidelities,
        env_energies=env_ev_energies,
        omegas=omegas,
        eigenspectrums=[
            sys_eig_energies - sys_eig_energies[0],
        ],
    )
    if save:
        fig.savefig("./fig8.png")
    plt.show()


def figure_10(save: bool = True):
    fig = compare_sweep_subspaces("slater", 6)
    if save:
        fig.savefig("./fig10.png")
    plt.show()


def figure_4(save: bool = True):
    fig = compare_sweep_subspaces("coulomb", 6)
    if save:
        fig.savefig("./fig4.png")
    plt.show()


def figure_9(save: bool = True):
    fig = compare_single_run()
    if save:
        fig.savefig("./fig9.png")
    plt.show()


def figure_5(save: bool = True):
    fig = thermalizer_run()
    if save:
        fig.savefig("./fig5.png")
    plt.show()


def figure_11(save: bool = True):
    fig = free_couplers_paulis(-1)
    if save:
        fig.savefig("./fig11.png")
    plt.show()


if __name__ == "__main__":
    save = False
    # figure_3(save) # spectroscopy run X
    # figure_4(save)  # coulomb comparison X
    # figure_5(save)  # thermalizing run X
    # figure_7(save) # fully noisy run X
    # figure_8(save) # parity conserving noisy run X
    # figure_9(save)  # single with/without sweep comparison X
    # figure_10(save)  # slater comparison X
    # figure_11(save)  # Sorted coefficients of the free couplers' Pauli strings
