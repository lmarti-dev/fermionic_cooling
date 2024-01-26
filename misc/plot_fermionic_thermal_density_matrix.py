import numpy as np
from openfermion import get_sparse_operator
from utils import (
    state_fidelity_to_eigenstates,
    thermal_density_matrix_at_particle_number,
)

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})

model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
n_electrons = [2, 2]

eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(model.fock_hamiltonian),
    particle_number=n_electrons,
)

n_steps = 100
n_dims = len(eigenstates[:, 0])

total_fids = np.zeros((n_dims, n_steps))

for ind, beta_power in enumerate(np.linspace(-2, 2, n_steps)):
    beta = 10**beta_power
    thermal_sys_density = thermal_density_matrix_at_particle_number(
        beta=beta,
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=False,
    )
    fidelities = state_fidelity_to_eigenstates(
        thermal_sys_density, eigenstates=eigenstates, expanded=False
    )
    total_fids[:, ind] = fidelities


fig, ax = plt.subplots()
cmap = plt.get_cmap("turbo", len(total_fids))
for ind, fids in enumerate(total_fids):
    ax.plot(
        range(len(fids)), np.abs(fids), label=rf"$|E_{{{ind}}}\rangle$", color=cmap(ind)
    )
ax.set_ylabel("Amplitude")
ax.set_xlabel(r"$\beta$")
ax.set_yscale("log")
ax.set_xscale("log")


plt.show()
