from qutlet.models import FermiHubbardModel
from qutlet.utilities import jw_eigenspectrum_at_particle_number
from fermionic_cooling.utils import state_fidelity_to_eigenstates
from openfermion import get_sparse_operator
import numpy as np


n_electrons = [2, 2]
model = FermiHubbardModel(
    lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
)
n_qubits = len(model.qubits)


cou_sys_eig_energies, cou_sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.coulomb_model.fock_hamiltonian,
        n_qubits=len(model.qubits),
    ),
    particle_number=n_electrons,
    expanded=True,
)
sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.fock_hamiltonian,
        n_qubits=len(model.qubits),
    ),
    particle_number=n_electrons,
    expanded=True,
)


eig_fids = state_fidelity_to_eigenstates(sys_eig_states[:, 0], cou_sys_eig_states)
print("fids")
for fid, ind in zip(eig_fids, range(cou_sys_eig_states.shape[1])):
    energy = model.hamiltonian.expectation_from_state_vector(
        cou_sys_eig_states[:, ind],
        qubit_map={k: n for n, k in enumerate(model.qubits)},
    )
    print(f"|E_{ind:0}>:fid: {np.abs(fid):.4f} E: {energy:.3f}")
print(f"sum fids {sum(eig_fids)}")
