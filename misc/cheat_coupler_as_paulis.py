from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from building_blocks import (
    get_Z_env,
    get_cheat_coupler_list,
)

from utils import ketbra
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, spin_dicke_state
from openfermion import get_sparse_operator, jw_hartree_fock_state
from data_manager import ExperimentDataManager


if __name__ == "__main__":
    data_folder = "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/data"
    dry_run = True

    edm = ExperimentDataManager(
        data_folder=data_folder,
        experiment_name="cheat_couplers_paulis",
        notes="checking the pauli form of the cheat couplers",
        dry_run=dry_run,
    )

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
    noise = 0
    n_env_qubits = 1
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    couplers = get_cheat_coupler_list(
        sys_eig_states=sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
        noise=noise,
        to_psum=True,
    )  # Interaction only on Qubit 0?

    print(couplers)

    jobj = {"couplers": couplers}

    edm.save_dict(jobj)
