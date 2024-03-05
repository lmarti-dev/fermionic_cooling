import sys

# tsk tsk
# sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from coolerClass import Cooler

from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_Z_env,
    get_cheat_coupler_list,
)

from utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
)
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
import matplotlib.pyplot as plt


from data_manager import ExperimentDataManager


def probe_reps_with_noise(
    edm: ExperimentDataManager,
    model: FermiHubbardModel,
    n_electrons: int,
    sys_qubits: list[cirq.Qid],
    sys_initial_state: np.ndarray,
    sys_eig_energies: np.ndarray,
    sys_eig_states: np.ndarray,
    sys_ground_state: np.ndarray,
    env_qubits: list[cirq.Qid],
    env_ground_state: np.ndarray,
    env_ham: cirq.PauliSum,
    env_eig_states: np.ndarray,
):
    noise_range = 10 ** np.linspace(-4, 0, 10)
    n_reps = 5
    end_fidelities = np.zeros((n_reps, len(noise_range)))
    for n_rep in range(n_reps):
        actual_rep = 4 * (n_rep + 1)
        for noise_ind, noise in enumerate(noise_range):
            print(f"noise: {noise}\n\n")
            couplers = get_cheat_coupler_list(
                sys_eig_states=sys_eig_states,
                env_eig_states=env_eig_states,
                qubits=sys_qubits + env_qubits,
                gs_indices=(0,),
                noise=noise,
            )  # Interaction only on Qubit 0?
            print("coupler done")

            print(f"number of couplers: {len(couplers)}")
            # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

            # get environment ham sweep values
            spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

            min_gap = sorted(np.abs(np.diff(sys_eig_energies)))[0]

            n_steps = len(couplers)
            # sweep_values = get_log_sweep(spectrum_width, n_steps)
            sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
            # np.random.shuffle(sweep_values)
            # coupling strength value
            alphas = sweep_values / 100
            evolution_times = 2.5 * np.pi / (alphas)
            # evolution_time = 1e-3

            # call cool

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

            # probe_times(edm, cooler, alphas, sweep_values)

            fidelities, energies, final_sys_density_matrix = cooler.zip_cool(
                alphas=alphas,
                evolution_times=evolution_times,
                sweep_values=sweep_values,
                n_rep=actual_rep,
            )

            jobj = {
                "fidelities": fidelities,
                "energies": energies,
            }
            edm.save_dict_to_experiment(
                filename=f"data_noise_{n_rep}_{noise:.4f}", jobj=jobj
            )

            end_fidelities[n_rep, noise_ind] = fidelities[-1]
    fig, ax = plt.subplots()
    for n_rep in range(n_reps):
        actual_rep = 4 * (n_rep + 1)
        ax.plot(
            noise_range,
            end_fidelities[n_rep, :],
            "x--",
            label=f"{actual_rep} rep.",
        )
    ax.set_xlabel("Noise coefficient [-]")
    ax.set_ylabel("Final fidelity")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("Effect of cooling repetitions on noisy coupler")

    plt.tight_layout()

    edm.save_figure(
        fig,
    )

    plt.show()


def probe_alpha_with_noise(
    edm: ExperimentDataManager,
    model: FermiHubbardModel,
    n_electrons: int,
    sys_qubits: list[cirq.Qid],
    sys_initial_state: np.ndarray,
    sys_eig_energies: np.ndarray,
    sys_eig_states: np.ndarray,
    sys_ground_state: np.ndarray,
    env_qubits: list[cirq.Qid],
    env_ground_state: np.ndarray,
    env_ham: cirq.PauliSum,
    env_eig_states: np.ndarray,
):
    noise_range = 10 ** np.linspace(-4, 1, 10)
    weaken_couplings = 10 ** np.arange(0, 5)
    end_fidelities = np.zeros((len(weaken_couplings), len(noise_range)))
    for wc_ind, weaken_coupling in enumerate(weaken_couplings):
        for noise_ind, noise in enumerate(noise_range):
            print(f"noise: {noise}\n\n")
            couplers = get_cheat_coupler_list(
                sys_eig_states=sys_eig_states,
                env_eig_states=env_eig_states,
                qubits=sys_qubits + env_qubits,
                gs_indices=(0,),
                noise=noise,
            )  # Interaction only on Qubit 0?
            print("coupler done")

            print(f"number of couplers: {len(couplers)}")
            # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

            # get environment ham sweep values
            spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

            min_gap = sorted(np.abs(np.diff(sys_eig_energies)))[0]

            n_steps = len(couplers)
            # sweep_values = get_log_sweep(spectrum_width, n_steps)
            sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
            # np.random.shuffle(sweep_values)
            # coupling strength value
            alphas = sweep_values / weaken_coupling
            evolution_times = 2.5 * np.pi / (alphas)
            # evolution_time = 1e-3

            # call cool

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

            # probe_times(edm, cooler, alphas, sweep_values)

            fidelities, energies, _ = cooler.zip_cool(
                alphas=alphas,
                evolution_times=evolution_times,
                sweep_values=sweep_values,
                n_rep=10,
            )

            jobj = {
                "noise": noise,
                "weaken_coupling": int(weaken_coupling),
                "fidelities": fidelities,
                "energies": energies,
            }
            edm.save_dict_to_experiment(
                filename=f"data_noise_{weaken_coupling:.3f}_{noise:.4f}", jobj=jobj
            )

            end_fidelities[wc_ind, noise_ind] = fidelities[-1]
    fig, ax = plt.subplots()
    for wc_ind, weaken_coupling in enumerate(weaken_couplings):
        ax.plot(
            noise_range,
            end_fidelities[wc_ind, :],
            "x-",
            label=rf"$\alpha/{weaken_coupling:.3f}$",
        )
    ax.set_xlabel("Noise coefficient [-]")
    ax.set_ylabel("Final fidelity")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()

    edm.save_figure(
        fig,
    )

    plt.show()


def probe_noise(
    edm: ExperimentDataManager,
    model: FermiHubbardModel,
    n_electrons: int,
    sys_qubits: list[cirq.Qid],
    sys_initial_state: np.ndarray,
    sys_eig_energies: np.ndarray,
    sys_eig_states: np.ndarray,
    sys_ground_state: np.ndarray,
    env_qubits: list[cirq.Qid],
    env_ground_state: np.ndarray,
    env_ham: cirq.PauliSum,
    env_eig_states: np.ndarray,
):
    noise_range = 10 ** np.linspace(-4, 1, 100)
    end_fidelities = np.zeros((len(noise_range),))
    weaken_coupling = 100
    for noise_ind, noise in enumerate(noise_range):
        print(f"noise: {noise}\n\n")
        couplers = get_cheat_coupler_list(
            sys_eig_states=sys_eig_states,
            env_eig_states=env_eig_states,
            qubits=sys_qubits + env_qubits,
            gs_indices=(0,),
            noise=noise,
        )  # Interaction only on Qubit 0?
        print("coupler done")

        print(f"number of couplers: {len(couplers)}")
        # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

        # get environment ham sweep values
        spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

        min_gap = sorted(np.abs(np.diff(sys_eig_energies)))[0]

        n_steps = len(couplers)
        # sweep_values = get_log_sweep(spectrum_width, n_steps)
        sweep_values = get_cheat_sweep(sys_eig_energies, n_steps)
        # np.random.shuffle(sweep_values)
        # coupling strength value
        alphas = sweep_values / weaken_coupling
        evolution_times = 2.5 * np.pi / (alphas)
        # evolution_time = 1e-3

        # call cool

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

        # probe_times(edm, cooler, alphas, sweep_values)

        fidelities, energies, _ = cooler.zip_cool(
            alphas=alphas,
            evolution_times=evolution_times,
            sweep_values=sweep_values,
            n_rep=1,
        )

        jobj = {
            "noise": noise,
            "weaken_coupling": int(weaken_coupling),
            "fidelities": fidelities,
            "energies": energies,
        }
        edm.save_dict_to_experiment(
            filename=f"data_noise_{weaken_coupling:.3f}_{noise:.4f}", jobj=jobj
        )

        end_fidelities[noise_ind] = fidelities[-1]
    fig, ax = plt.subplots()
    ax.plot(
        noise_range,
        end_fidelities,
        "x-",
        markersize=5,
    )
    ax.set_xlabel("Noise coefficient [-]")
    ax.set_ylabel("Final fidelity")
    ax.set_xscale("log")
    # ax.legend()
    plt.tight_layout()

    edm.save_figure(
        fig,
    )

    plt.show()


def __main__(args):
    # whether we want to skip all saving data
    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="cooling_check_noise_vs_reps",
        notes="trying out the effect of noise on cheat couplers",
        dry_run=dry_run,
    )
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 1]
    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=True
    )
    sys_initial_state = ketbra(sys_hartree_fock)
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_state = sys_eig_states[:, np.argmin(sys_eig_energies)]
    sys_ground_energy = np.min(sys_eig_energies)

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
        sys_eigenspectrum=sys_eig_energies,
        env_eigenergies=env_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )
    probe_noise(
        edm,
        model,
        n_electrons,
        sys_qubits,
        sys_initial_state,
        sys_eig_energies,
        sys_eig_states,
        sys_ground_state,
        env_qubits,
        env_ground_state,
        env_ham,
        env_eig_states,
    )


if __name__ == "__main__":
    __main__(sys.argv)
