from fauvqe.models import FermiHubbardModel
from scipy.linalg import expm
import numpy as np

model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)

n_electrons = [2, 2]

beta = 10

thermal_sys_density = expm(
    -beta * model.hamiltonian.matrix(qubits=model.flattened_qubits)
)
thermal_sys_density /= np.trace(thermal_sys_density)
model.fermionic_spin_and_number()
