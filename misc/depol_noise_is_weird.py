from fermionic_cooling.utils import add_depol_noise, ketbra

from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from openfermion import get_sparse_operator
import numpy as np

x = 2
y = 2
n_electrons = [2, 2]

depol_noise = 1

model = FermiHubbardModel(x_dimension=x, y_dimension=y, tunneling=1, coulomb=2)

eig_energies, eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(model.fock_hamiltonian),
    particle_number=n_electrons,
    expanded=True,
)

rho = ketbra(eig_states[:, 2])

n_up, n_down = model.fermion_spins_expectations(rho)
print(f"Before: up: {n_up:.4f} down: {n_down:.4f}")

noisy = add_depol_noise(
    rho=rho,
    depol_noise=depol_noise,
    n_qubits=len(model.flattened_qubits),
    n_electrons=n_electrons,
    is_noise_spin_conserving=False,
)

n_up, n_down = model.fermion_spins_expectations(noisy)
print(f"After: up: {n_up:.4f} down: {n_down:.4f}")

print(rho.shape)
print(noisy.shape)

print(f"Diff: {np.sum(rho-noisy)}")

# the expectation value of the maxmimally mixed state is half-filling, but with everything else in it
# can we do something about this
