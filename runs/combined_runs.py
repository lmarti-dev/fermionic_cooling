# there are five cases
# 1. sweep only (done)
# 2. cooling only (done)
# 3. cooling then sweep
# 4. sweep then cooling
# 5. sweep and cooling
from adiabatic_sweep import (
    run_sweep,
    get_sweep_hamiltonian,
    fermion_to_dense,
)
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from utils import (
    get_extrapolated_superposition,
    get_slater_spectrum,
    ketbra,
    trace_out_env,
    get_min_gap,
)
from coolerClass import Cooler
import numpy as np
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
)

import matplotlib.pyplot as plt
from openfermion import get_sparse_operator, jw_hartree_fock_state

from building_blocks import (
    get_Z_env,
    get_cheat_coupler_list,
    control_function,
)


def plot_combined_run(
    first_fidelities: list,
    first_time: float,
    second_fidelities: list,
    second_time: float,
):
    fig, ax = plt.subplots()
    first_steps = np.linspace(0, first_time, len(first_fidelities))
    second_steps = np.linspace(
        first_time, first_time + second_time, len(second_fidelities)
    )

    ax.plot(first_steps, first_fidelities, "r")
    ax.plot(second_steps, second_fidelities, "b")
    total_fidelities = first_fidelities + second_fidelities
    ax.vlines(
        first_time,
        ymin=np.min(total_fidelities),
        ymax=np.max(total_fidelities),
        linestyles="dotted",
        colors="k",
        label="procedure switch",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    plt.show()


def cooling_then_sweep(
    cooler: Cooler,
    model: FermiHubbardModel,
    n_electrons: list,
    n_cooling_steps: int,
    cooling_time: float,
    n_sweep_steps: int,
    sweep_time: float,
):
    cooling_fidelities = []
    weaken_coupling = 100
    n_qubits = len(model.flattened_qubits)

    ham_start = fermion_to_dense(model.non_interacting_model.fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)

    slater_energies, _ = get_slater_spectrum(model=model, n_electrons=n_electrons)
    omega = 1.02 * np.abs(np.max(slater_energies) - np.min(slater_energies))
    total_density_matrix = cooler.total_initial_state

    print("cooling")
    for ind, evolution_time in enumerate(np.linspace(0, cooling_time, n_cooling_steps)):
        print(f"step {ind} time: {evolution_time:.3f} omega {omega:.3f}", end="\r")
        cooler.sys_env_coupler_easy_setter(ind, 0)
        alpha = omega / (weaken_coupling * n_qubits)
        (
            sys_fidelity,
            sys_energy,
            env_energy,
            total_density_matrix,
        ) = cooler.cooling_step(
            total_density_matrix=total_density_matrix,
            env_coupling=omega,
            alpha=alpha,
            evolution_time=evolution_time,
        )
        cooling_fidelities.append(sys_fidelity)
        omega = omega - control_function(
            omega, t_fridge=env_energy, beta=1, mu=20, c=10
        )
    sys_density_matrix = trace_out_env(
        total_density_matrix,
        n_sys_qubits=len(cooler.sys_qubits),
        n_env_qubits=len(cooler.env_qubits),
    )

    print(f"final fid {cooling_fidelities[-1]:.3f}")
    print("sweeping")
    sweep_fidelities, sweep_instant_fidelities, final_ground_state = run_sweep(
        initial_state=sys_density_matrix,
        ham_start=ham_start,
        ham_stop=ham_stop,
        n_electrons=n_electrons,
        n_steps=n_sweep_steps,
        total_time=sweep_time,
    )

    print(f"final fid {sweep_fidelities[-1]:.3f}")
    plot_combined_run(
        first_fidelities=cooling_fidelities,
        first_time=cooling_time,
        second_fidelities=sweep_fidelities,
        second_time=sweep_time,
    )


def sweep_then_cooling(
    cooler: Cooler,
    model: FermiHubbardModel,
    n_electrons: list,
    n_cooling_steps: int,
    cooling_time: float,
    n_sweep_steps: int,
    sweep_time: float,
):
    cooling_fidelities = []
    weaken_coupling = 100
    n_qubits = len(model.flattened_qubits)

    ham_start = fermion_to_dense(model.non_interacting_model.fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)
    slater_energies, _ = get_slater_spectrum(model=model, n_electrons=n_electrons)

    initial_state = get_extrapolated_superposition(
        model, n_electrons=n_electrons, coulomb=1e-6
    )
    print("sweeping")
    sweep_fidelities, sweep_instant_fidelities, final_ground_state = run_sweep(
        initial_state=initial_state,
        ham_start=ham_start,
        ham_stop=ham_stop,
        n_electrons=n_electrons,
        n_steps=n_sweep_steps,
        total_time=sweep_time,
    )
    print(f"final fid {sweep_fidelities[-1]:.3f}")

    # the initial state is the cooler is final x ground env
    cooler.sys_initial_state = ketbra(final_ground_state)

    # make sure you only cool the first coupler
    cooler.sys_env_coupler_data = cooler.sys_env_coupler_data[0]
    eigenenergies, _ = jw_eigenspectrum_at_particle_number(
        get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    omega = get_min_gap(eigenenergies, threshold=1e-8)

    print("cooling")
    total_density_matrix = cooler.total_initial_state
    for ind, evolution_time in enumerate(np.linspace(0, cooling_time, n_cooling_steps)):
        print(f"step {ind} time: {evolution_time:.3f} omega {omega:.3f}", end="\r")
        cooler.sys_env_coupler_easy_setter(ind, 0)
        alpha = omega / (weaken_coupling * n_qubits)
        (
            sys_fidelity,
            sys_energy,
            env_energy,
            total_density_matrix,
        ) = cooler.cooling_step(
            total_density_matrix=total_density_matrix,
            env_coupling=omega,
            alpha=alpha,
            evolution_time=evolution_time,
        )
        cooling_fidelities.append(sys_fidelity)
    print(f"final fid {cooling_fidelities[-1]:.3f}")

    plot_combined_run(
        second_fidelities=cooling_fidelities,
        second_time=cooling_time,
        first_fidelities=sweep_fidelities,
        first_time=sweep_time,
    )


def combined_sweep_cooling():
    pass


def __main__():
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)

    n_electrons = [2, 2]
    n_env_qubits = 1
    n_sys_qubits = len(model.flattened_qubits)
    total_time = 130
    n_cooling_steps = 10
    n_sweep_steps = 100

    sys_qubits = model.flattened_qubits
    slater_energies, slater_eigenstates = get_slater_spectrum(
        model, n_electrons=n_electrons
    )
    sys_ground_energy, sys_ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )
    couplers = get_cheat_coupler_list(
        sys_eig_states=slater_eigenstates,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
        noise=0,
    )  # Interaction only on Qubit 0?$

    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )

    sys_initial_state = ketbra(sys_hartree_fock)
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
    # cooling_then_sweep(
    #     cooler=cooler,
    #     model=model,
    #     n_electrons=n_electrons,
    #     n_cooling_steps=n_cooling_steps,
    #     cooling_time=total_time / 2,
    #     n_sweep_steps=n_sweep_steps,
    #     sweep_time=total_time / 2,
    # )
    sweep_then_cooling(
        cooler=cooler,
        model=model,
        n_electrons=n_electrons,
        n_cooling_steps=n_cooling_steps,
        cooling_time=total_time / 2,
        n_sweep_steps=n_sweep_steps,
        sweep_time=total_time / 2,
    )


if __name__ == "__main__":
    __main__()
