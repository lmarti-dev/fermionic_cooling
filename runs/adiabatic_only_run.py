from adiabatic_sweep import (
    run_sweep,
    get_sweep_hamiltonian,
    fermion_to_dense,
    get_instantaneous_ground_states,
)
from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from data_manager import ExperimentDataManager

from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
)
from utils import (
    get_closest_noninteracting_degenerate_ground_state,
    get_min_gap,
    extrapolate_ground_state_non_interacting_fermi_hubbard,
    get_extrapolated_superposition,
)

from openfermion import get_sparse_operator, jw_hartree_fock_state
import matplotlib.pyplot as plt

from cirq import fidelity
import numpy as np


def plot_fidelity(fidelities, instant_fidelities):
    fig, ax = plt.subplots()

    ax.plot(range(len(fidelities)), fidelities, label="g.s.")
    ax.plot(range(len(instant_fidelities)), instant_fidelities, label="instant. g.s.")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Fidelity")
    ax.legend()

    plt.show()


def __main__():
    data_folder = "C:/Users/Moi4/Desktop/current/FAU/phd/data"

    # whether we want to skip all saving data
    dry_run = True
    edm = ExperimentDataManager(
        data_folder=data_folder,
        experiment_name="adiabatic_sweep_only",
        notes="trying out adiabatic sweep for fermi hubbard",
        dry_run=dry_run,
    )
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 2]

    n_qubits = len(model.flattened_qubits)

    extrapolated_ground_state = extrapolate_ground_state_non_interacting_fermi_hubbard(
        model=model, n_electrons=n_electrons, n_points=5, deg=1
    )

    eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    slater_eigenenergies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.non_interacting_model.fock_hamiltonian
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    extrapolated_superposition = get_extrapolated_superposition(
        model=model, n_electrons=n_electrons, coulomb=1e-6
    )

    final_ground_state = eigenstates[:, 0]
    initial_ground_state = slater_eigenstates[:, 2]
    print(
        f"initial fidelity: {fidelity(initial_ground_state,final_ground_state,qid_shape=(2,)*n_qubits)}"
    )

    ham_start = fermion_to_dense(model.non_interacting_model.fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)

    # total steps
    n_steps = int(1e2)
    # total time
    spectrum_width = np.max(eigenenergies) - np.min(eigenenergies)
    total_time = (
        10 * spectrum_width / (get_min_gap(eigenenergies, threshold=1e-12) ** 2)
    )

    instantaneous_ground_states = get_instantaneous_ground_states(
        ham_start=ham_start, ham_stop=ham_stop, n_steps=n_steps, n_electrons=n_electrons
    )

    print(f"Simulating for {total_time} time and {n_steps} steps")
    fidelities, instant_fidelities, final_ground_state, populations = run_sweep(
        initial_state=initial_ground_state,
        ham_start=ham_start,
        ham_stop=ham_stop,
        final_ground_state=final_ground_state,
        instantaneous_ground_states=instantaneous_ground_states,
        n_steps=n_steps,
        total_time=total_time,
        get_populations=True,
    )

    plt.rcParams.update(
        {
            "font.family": r"serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "figure.figsize": (5, 3),
        }
    )

    # pops
    fig, ax = plt.subplots()
    for pop in populations:
        ax.plot(range(len(pop)), pop, linewidth=0.5)
    ax.plot(range(len(fidelities)), fidelities, "k--", linewidth=3, label="Fidelity")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Fidelity")
    ax.set_yscale("log")
    ax.set_ybound(1e-7, 1)

    plt.show()


if __name__ == "__main__":
    __main__()
