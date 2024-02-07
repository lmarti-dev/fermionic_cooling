import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from helpers.specificModel import SpecificModel
from plotting.plot_comparison_adiabatic_preprocessing import (
    plot_results,
)

from coolerClass import Cooler
from adiabatic_sweep import run_sweep, fermion_to_dense
from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_perturbed_sweep,
    get_Z_env,
    get_cheat_coupler_list,
)

from utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
    get_min_gap,
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


from data_manager import ExperimentDataManager, set_color_cycler
import numpy as np


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="cooling_with_initial_adiab_sweep",
        notes="adding an initial sweep before the cooling run",
        dry_run=dry_run,
    )
    set_color_cycler()
    # model stuff

    model_name = "fh_coulomb"
    if "fh_" in model_name:
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
            couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
        elif "nonint" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            couplers_fock_hamiltonian = start_fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        start_fock_hamiltonian = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            couplers_fock_hamiltonian,
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
    start_eig_energies, start_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    sys_slater_state = start_eig_states[:, 0]
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

    edm.dump_some_variables(
        n_electrons=n_electrons,
        n_sys_qubits=n_sys_qubits,
        n_env_qubits=n_env_qubits,
        sys_eig_energies=sys_eig_energies,
        env_eig_energies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )

    couplers = get_cheat_coupler_list(
        sys_eig_states=free_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
        noise=0,
    )  # Interaction only on Qubit 0?
    print("coupler done")

    print(f"number of couplers: {len(couplers)}")
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

    min_gap = get_min_gap(free_sys_eig_energies, threshold=1e-6)

    # evolution_time = 1e-3

    fpaths = []
    for which_initial_process in ("adiabatic", "none"):
        print(f"Initial process: {which_initial_process}")

        if which_initial_process == "adiabatic":
            # call sweep
            initial_ground_state = sys_slater_state
            final_ground_state = sys_eig_states[:, 0]
            ham_start = fermion_to_dense(start_fock_hamiltonian)
            ham_stop = fermion_to_dense(model.fock_hamiltonian)
            n_steps = 100
            total_time = (
                0.001
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
                total_time=total_time,
                get_populations=True,
            )
            sys_initial_state = final_state
        elif which_initial_process == "none":
            sys_initial_state = sys_slater_state
        cooler = Cooler(
            sys_hamiltonian=model.hamiltonian,
            n_electrons=n_electrons,
            sys_qubits=model.flattened_qubits,
            sys_ground_state=sys_ground_state,
            sys_initial_state=sys_initial_state,
            env_hamiltonian=env_ham,
            env_qubits=env_qubits,
            env_ground_state=env_ground_state,
            sys_env_coupler_data=couplers,
            verbosity=5,
        )
        n_rep = 1

        print(f"coupler dim: {cooler.sys_env_coupler_data_dims}")

        ansatz_options = {"beta": 1, "mu": 20, "c": 10}
        weaken_coupling = 30

        start_omega = 3

        stop_omega = 0.1

        fidelities, sys_ev_energies, omegas, env_ev_energies = cooler.big_brain_cool(
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
            "env_ev_energies": env_ev_energies,
        }
        fpaths.append(
            edm.save_dict_to_experiment(
                filename=f"cooling_free_{which_initial_process}",
                jobj=jobj,
                return_fpath=True,
            )
        )

    plot_results(edm, fpaths[0], fpaths[1], sys_eig_energies)


if __name__ == "__main__":
    __main__(sys.argv)
