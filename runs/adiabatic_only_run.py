import matplotlib.pyplot as plt
import numpy as np
from adiabatic_sweep import (
    fermion_to_dense,
    get_instantaneous_ground_states,
    get_sweep_hamiltonian,
    run_sweep,
)
from cirq import fidelity
from helpers.specificModel import SpecificModel
from openfermion import (
    get_sparse_operator,
    jw_hartree_fock_state,
    get_quadratic_hamiltonian,
)
from utils import (
    extrapolate_ground_state_non_interacting_fermi_hubbard,
    get_closest_noninteracting_degenerate_ground_state,
    get_extrapolated_superposition,
    get_min_gap,
    state_fidelity_to_eigenstates,
)

from data_manager import ExperimentDataManager
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
)


def plot_fidelity(fidelities, instant_fidelities):
    fig, ax = plt.subplots()

    ax.plot(range(len(fidelities)), fidelities, label="g.s.")
    ax.plot(range(len(instant_fidelities)), instant_fidelities, label="instant. g.s.")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Fidelity")
    ax.legend()

    # plt.show()


def __main__():
    model_names = (
        "v3/FAU_O2_singlet_6e_4o_CASSCF",
        "v3/FAU_O2_singlet_8e_6o_CASSCF",
        "v3/Fe3_NTA_quartet_CASSCF",
        "v3/FAU_O2_triplet_6e_4o_CASSCF",
    )
    dry_run = False

    edm = ExperimentDataManager(
        experiment_name="adiabatic_sweep_chems",
        notes="adiabatic sweep for chemical models",
        dry_run=dry_run,
    )
    for model_name in model_names:
        run_comp(edm, model_name)
        edm.new_run()


def run_comp(edm, model_name):
    # whether we want to skip all saving data

    # model stuff
    if model_name == "fh":
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        non_interacting_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        non_interacting_fock_hamiltonian = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    # extrapolated_ground_state = extrapolate_ground_state_non_interacting_fermi_hubbard(
    #     model=model, n_electrons=n_electrons, n_points=5, deg=1
    # )

    # extrapolated_superposition = get_extrapolated_superposition(
    #     model=model, n_electrons=n_electrons, coulomb=1e-6
    # )

    eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    slater_eigenenergies, slater_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(non_interacting_fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    final_ground_state = eigenstates[:, 0]
    initial_ground_state = slater_eigenstates[:, 0]
    print(
        f"initial fidelity: {fidelity(initial_ground_state,final_ground_state,qid_shape=(2,)*n_qubits)}"
    )

    ham_start = fermion_to_dense(non_interacting_fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)

    # total steps
    n_steps = int(1e2)
    # total time
    spectrum_width = np.max(eigenenergies) - np.min(eigenenergies)
    total_time = spectrum_width / (get_min_gap(eigenenergies, threshold=1e-6) ** 2)

    edm.dump_some_variables(
        model_name=model_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        spectrum_width=spectrum_width,
        total_time=total_time,
    )
    instantaneous_ground_states = get_instantaneous_ground_states(
        ham_start=ham_start, ham_stop=ham_stop, n_steps=n_steps, n_electrons=n_electrons
    )

    print(f"Simulating for {total_time} time and {n_steps} steps")
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
        instantaneous_ground_states=instantaneous_ground_states,
        n_steps=n_steps,
        total_time=total_time,
        get_populations=True,
    )

    fid_init = state_fidelity_to_eigenstates(initial_ground_state, eigenstates)
    fid_final = state_fidelity_to_eigenstates(final_state, eigenstates)

    print("=============")

    for ind, (a, b) in enumerate(zip(fid_init, fid_final)):
        print(
            f"E_{ind} init pop: {a:.3f} dE: {slater_eigenenergies[ind]-slater_eigenenergies[0]:.3f} final pop {b:.3f} dE: {eigenenergies[ind]-eigenenergies[0]:.3f}"
        )

    # pops
    fig, ax = plt.subplots()
    plot_populations = False
    if plot_populations:
        for pop in populations:
            ax.plot(range(len(pop)), pop, linewidth=0.5)
    ax.plot(
        np.linspace(0, total_time, len(fidelities)),
        fidelities,
        "r",
        linewidth=2,
        label="Fidelity to g.s.",
    )
    ax.plot(
        np.linspace(0, total_time, len(instant_fidelities)),
        instant_fidelities,
        "b",
        linewidth=2,
        label="Fidelity to instant g.s.",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    ax.legend()
    # ax.set_yscale("log")
    # ax.set_ybound(1e-7, 1)

    edm.save_figure(fig=fig)


if __name__ == "__main__":
    __main__()
