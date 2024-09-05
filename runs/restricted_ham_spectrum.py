from qutlet.models import RandomFermionicModel
from fermionic_cooling.utils import dense_restricted_ham
import numpy as np

model = RandomFermionicModel(8, n_electrons="half-filling")

sys_ham_matrix = dense_restricted_ham(
    model.fock_hamiltonian, model.n_electrons, model.n_qubits
)

ss_ene, ss_states = model.subspace_spectrum

dr_ene, dr_states = np.linalg.eigh(sys_ham_matrix)


assert np.all(np.isclose(ss_ene, dr_ene))
assert np.all(np.isclose(ss_states, dr_states))
