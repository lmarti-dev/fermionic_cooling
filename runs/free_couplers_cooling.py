import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from chemical_models.specificModel import SpecificModel

from coolerClass import Cooler

from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_perturbed_sweep,
    get_Z_env,
    get_cheat_coupler_list,
)

from fermionic_cooling.utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
    get_min_gap,
    s_squared_penalty,
)
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    spin_dicke_state,
    flatten,
)
import cirq
from openfermion import (
    get_sparse_operator,
    jw_hartree_fock_state,
    get_quadratic_hamiltonian,
)
import matplotlib.pyplot as plt


from fermionic_cooling.adiabatic_sweep import fermion_to_dense, run_sweep

from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style
import numpy as np


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="fh_new_freecouplers",
        notes="cooling this home cooked chemical ham with free couplers",
        dry_run=dry_run,
    )
    use_style()
    # model stuff

    model_name = "cooked/SingletO2_6e_8q"
    model_name = "fh"
    if model_name == "fh":
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        non_interacting_model = model.non_interacting_model.fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        non_interacting_model = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    gs_index = 2
    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            non_interacting_model,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )

    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=False
    )
    sys_mixed_state = np.ones(2**n_sys_qubits) / (2 ** (n_sys_qubits / 2))
    sys_slater_state = free_sys_eig_states[:, gs_index]

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    # initial state setting
    sys_initial_state = ketbra(sys_slater_state)

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

    eig_fids = state_fidelity_to_eigenstates(
        state=sys_initial_state, eigenstates=sys_eig_states
    )
    print("Initial populations")
    for fid, sys_eig_energy in zip(eig_fids, sys_eig_energies):
        print(
            f"fid: {np.abs(fid):.4f} gap: {np.abs(sys_eig_energy-sys_eig_energies[0]):.3f}"
        )
    print(f"sum fids {sum(eig_fids)}")
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
    use_fast_sweep = False
    depol_noise = 1e-4
    is_noise_spin_conserving = False

    n_up, n_down = model.fermion_spins_expectations(sys_initial_state)
    print(f"Before fast sweep: up: {n_up:.4f} down: {n_down:.4f}")

    if use_fast_sweep:
        sweep_time_mult = 0.01
        start_fock_hamiltonian = non_interacting_model
        initial_ground_state = ketbra(sys_slater_state)
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
            n_electrons=n_electrons,
        )
        sys_initial_state = final_state
    else:
        total_sweep_time = 0

    penalty = s_squared_penalty(n_qubits=n_sys_qubits, n_electrons=n_electrons)
    use_penalty = False
    if use_penalty:
        jw_penalty = model.encode_as_self(penalty)
        total_ham = model.hamiltonian + jw_penalty
    else:
        total_ham = model.hamiltonian

    n_up, n_down = model.fermion_spins_expectations(sys_initial_state)
    print(f"After fast sweep: up: {n_up:.4f} down: n_down: {n_down:.4f}")

    cooler = Cooler(
        sys_hamiltonian=total_ham,
        n_electrons=n_electrons,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
        time_evolve_method="diag",
    )
    n_rep = 1

    print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

    ansatz_options = {"beta": 1, "mu": 30, "c": 20}
    weaken_coupling = 100

    start_omega = 1.75
    stop_omega = 0.3

    method = "bigbrain"

    edm.var_dump(
        depol_noise=depol_noise,
        use_fast_sweep=use_fast_sweep,
        is_noise_spin_conserving=is_noise_spin_conserving,
        max_k=max_k,
        ansatz_options=ansatz_options,
        use_penalty=use_penalty,
        method=method,
    )

    if method == "bigbrain":
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
        }
        edm.save_dict(filename="cooling_free", jobj=jobj)

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

    elif method == "zipcool":
        initial_pops = state_fidelity_to_eigenstates(
            state=sys_initial_state, eigenstates=sys_eig_states
        )
        n_rep = 1
        sweep_values = np.array(
            list(
                flatten(
                    get_perturbed_sweep(free_sys_eig_energies, x)
                    for x in np.linspace(2.5, 3.5, 30)
                )
            )
        )

        sweep_values = get_cheat_sweep(spectrum=sys_eig_energies)

        # sweep_values = get_cheat_sweep(sys_eig_energies, n_steps=len(sys_eig_energies))
        alphas = sweep_values / 50
        evolution_times = 2.5 * np.pi / (alphas)
        (
            fidelities,
            sys_energies,
            env_energies,
            final_sys_density_matrix,
        ) = cooler.zip_cool(
            alphas=alphas,
            evolution_times=evolution_times,
            sweep_values=sweep_values,
            n_rep=n_rep,
            fidelity_threshold=2,
        )

        jobj = {
            "fidelities": fidelities,
            "sys_energies": sys_energies,
            "env_energies": env_energies,
            "total_sweep_time": total_sweep_time,
            "total_cooling_time": cooler.total_cooling_time,
        }
        edm.save_dict(filename="cooling_free_couplers", jobj=jobj)

        fig, ax = plt.subplots()

        ax.plot(sweep_values, fidelities[1:], "k--")
        ax.set_xlabel("$\omega$")
        ax.set_ylabel("Fidelity")
        ax.invert_xaxis()

        edm.save_figure(
            fig,
        )

        plt.show()


if __name__ == "__main__":
    __main__(sys.argv)
