from qutlet.models import RandomFermionicModel

from qutlet.utilities import jw_hartree_fock_state
from fermionic_cooling.utils import subspace_energy_expectation, expectation_wrapper


model = RandomFermionicModel(
    8, init_coefficients=("r[-1,-.01]", "r[0.01,6]"), n_electrons="hf"
)


energies, eigenstates = model.subspace_spectrum


print(energies)


print(subspace_energy_expectation(eigenstates[:, 0], energies, eigenstates))
print(expectation_wrapper(model.hamiltonian, eigenstates[:, 0], qubits=model.qubits))
