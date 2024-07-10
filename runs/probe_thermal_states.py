from qutlet.models import FermiHubbardModel
from scipy.linalg import expm
import numpy as np

n_electrons = [2, 2]

model = FermiHubbardModel(
    lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
)

beta = 10

thermal_sys_density = expm(-beta * model.hamiltonian.matrix(qubits=model.qubits))
thermal_sys_density /= np.trace(thermal_sys_density)
model.fermionic_spin_and_number()
