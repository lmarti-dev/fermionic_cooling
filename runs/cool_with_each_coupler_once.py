import sys
from itertools import combinations

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
    spin_dicke_state,
)
from fermionic_cooling.utils import (
    expectation_wrapper,
    get_min_gap,
    ketbra,
    print_state_fidelity_to_eigenstates,
    state_fidelity_to_eigenstates,
    dense_restricted_ham,
    subspace_energy_expectation,
)

from adiabatic_sweep import run_sweep


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="fh_bigbrain_subspace",
        notes="fh cooling with subspace simulation, with coulomb",
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

    coupler_sys_eig_energies, couplers_sys_eig_states = (
        jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )
    )
    start_sys_eig_energies, start_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
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

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    # initial state setting
    gs_index = 2
    sys_initial_state = ketbra(start_sys_eig_states[:, gs_index])

    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

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

    max_k = None

    coupler_gs_index = 2
    couplers = get_cheat_coupler_list(
        sys_eig_states=couplers_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(coupler_gs_index,),
        noise=None,
        max_k=max_k,
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(coupler_sys_eig_energies, threshold=1e-6)

    # evolution_time = 1e-3

    # call cool
    use_fast_sweep = True
    depol_noise = None
    is_noise_spin_conserving = False

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

    for ind, coupler in enumerate(couplers):
        if ind > 0:
            edm.new_run(notes=f"coupler ({ind+1},0)")

        cooler = Cooler(
            sys_hamiltonian=sys_ham_matrix,
            n_electrons=n_electrons,
            sys_qubits=model.flattened_qubits,
            sys_ground_state=sys_ground_state,
            sys_initial_state=sys_initial_state,
            env_hamiltonian=env_ham,
            env_qubits=env_qubits,
            env_ground_state=env_ground_state,
            sys_env_coupler_data=[coupler],
            verbosity=5,
            time_evolve_method="expm",
            subspace_simulation=True,
        )
        n_rep = 1

        print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

        ansatz_options = {"beta": 1, "mu": 20, "c": 40}
        weaken_coupling = 20

        start_omega = 1.01 * spectrum_width

        stop_omega = 0.1

        method = "bigbrain"

        if method == "bigbrain":

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

            print(f"coupler number: {ind} final fidelity: {fidelities[0][-1]}")
            # plt.show()
            edm.var_dump(
                coupler_number=ind,
                final_fidelity=fidelities[0][-1],
                delta_fidelity=fidelities[0][-1] - fidelities[0][0],
            )


if __name__ == "__main__":
    __main__(sys.argv)
