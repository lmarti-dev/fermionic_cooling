from qutlet.models import FermiHubbardModel
from qutlet.utilities import jw_eigenspectrum_at_particle_number
from openfermion import get_sparse_operator


from itertools import combinations
from fermionic_cooling.utils import dense_restricted_ham

if __name__ == "__main__":
    model_name = "fh_coulomb"
    if "fh_" in model_name:
        n_electrons = [2, 2]
        model = FermiHubbardModel(
            lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
        )
        n_qubits = len(model.qubits)
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
            couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
        elif "slater" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            couplers_fock_hamiltonian = start_fock_hamiltonian

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )

    n_sys_qubits = len(model.qubits)

    subspace_dim = len(
        list(combinations(range(n_sys_qubits // 2), n_electrons[0]))
    ) * len(list(combinations(range(n_sys_qubits // 2), n_electrons[1])))

    sys_ham_matrix = dense_restricted_ham(
        model.fock_hamiltonian, n_electrons, n_sys_qubits
    )

    gs_index = 2
    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            start_fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=False,
    )
    print(free_sys_eig_energies)
    print(free_sys_eig_states[:, 1])
