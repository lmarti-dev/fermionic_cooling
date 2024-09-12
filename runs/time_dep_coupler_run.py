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
)


def get_ding_filter_function(a: float, da: float, b: float, db: float):
    # Single-ancilla ground state preparation via Lindbladians
    # paper values
    # a=2.5 * spectrum_width, da=0.5 * spectrum_width, b=min_gap (spectral gap), db=min_gap
    def filter_function(time: float):
        pa = np.exp((-(time**2) * (da**2) / 4)) * np.exp(1j * a * time)
        pb = np.exp((-(time**2) * (db**2) / 4)) * np.exp(1j * b * time)
        return (pa - pb) / (2 * np.pi * 1j * time)

    return filter_function


def get_lloyd_filter_function(biga: float, beta: float, tau: float):
    # Quasiparticle cooling algorithms for quantum many-body state preparation
    # arXiv:2404.12175v1 [quant-ph] 18 Apr 2024
    def filter_function(time: float):
        return np.sin((np.pi(time - tau) / 2)) / (
            (beta) * (np.sinh(np.pi(time - tau) / beta))
        )

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

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    subspace_simulation = True

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    _, start_gs_index = get_closest_state(
        ref_state=sys_ground_state,
        comp_states=start_sys_eig_states,
        subspace_simulation=subspace_simulation,
    )
    coupler_gs_index = start_gs_index

    # initial state setting
    # sys_initial_state = ketbra(start_sys_eig_states[:, start_gs_index])
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
    )

    max_k = None

    couplers = get_XsXbYsYb_coupler(
        sys_qubits=sys_qubits, env_qubits=env_qubits
    )  # Interaction only on Qubit 0?
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

    filter_function = get_ding_filter_function(
        a=2.5 * spectrum_width, da=0.5 * spectrum_width, b=min_gap, db=min_gap
    )
    times = np.linspace(0.01, 1, 100)
    (plot_times, fidelities, sys_ev_energies, env_ev_energies, total_density_matrix) = (
        cooler.time_cool(
            filter_function=filter_function, times=times, env_coupling=1.2, alpha=1
        )
    )

    jobj = {
        "fidelities": fidelities,
        "sys_energies": sys_ev_energies,
        "env_energies": env_ev_energies,
    }
    edm.save_dict(
        filename=f"cooling_free_{start_gs_index}_{coupler_gs_index}", jobj=jobj
    )

    fig = cooler.plot_time_cooling(plot_times, fidelities, env_ev_energies)
    edm.save_figure(
        fig,
    )
    plt.show()


if __name__ == "__main__":
    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        experiment_name="bigbrain_subspace_fh_x_gate",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    style()
    __main__(edm)
    # model stuff
