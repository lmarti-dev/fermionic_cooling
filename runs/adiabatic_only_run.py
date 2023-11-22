from adiabatic_sweep import (
    run_sweep,
    get_sweep_hamiltonian,
    fermion_to_dense,
)
from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from data_manager import ExperimentDataManager

from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    spin_dicke_state,
    jw_get_true_ground_state_at_particle_number,
)
from helpers.qubit_tools import (
    get_closest_quadratic_degenerate_ground_state,
)

from openfermion import get_sparse_operator, jw_hartree_fock_state
import matplotlib.pyplot as plt

from cirq import fidelity


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
    n_electrons = [2, 1]

    qubits = model.flattened_qubits
    n_qubits = len(qubits)
    hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_qubits, n_electrons=sum(n_electrons)
    )

    slater_energy, slater_state = get_closest_quadratic_degenerate_ground_state(
        fermion_operator=model.fock_hamiltonian, n_qubits=n_qubits, Nf=n_electrons
    )
    dicke_state = spin_dicke_state(
        n_qubits=n_qubits, Nf=n_electrons, right_to_left=False
    )

    ground_energy, ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian, n_qubits=n_qubits),
        particle_number=n_electrons,
    )

    initial_state = slater_state
    print(
        f"initial fidelity: {fidelity(slater_state,ground_state,qid_shape=(2,)*n_qubits)}"
    )

    ham_start = fermion_to_dense(model.non_interacting_model.fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)
    n_steps = 1000
    total_time = 10

    fidelities, instant_fidelities = run_sweep(
        initial_state=initial_state,
        ham_start=ham_start,
        ham_stop=ham_stop,
        n_electrons=n_electrons,
        n_steps=n_steps,
        total_time=total_time,
    )

    plt.rcParams.update(
        {
            "font.family": r"serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "figure.figsize": (5, 3),
        }
    )

    fig, ax = plt.subplots()

    ax.plot(range(len(fidelities)), fidelities, label="fid. to gs")
    ax.plot(
        range(len(instant_fidelities)), instant_fidelities, label="fid. to instant. gs"
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Fidelity")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    __main__()
