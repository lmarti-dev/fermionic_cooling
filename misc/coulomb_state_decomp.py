from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number, qmap
from fermionic_cooling.utils import state_fidelity_to_eigenstates
from openfermion import get_sparse_operator
import numpy as np


model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
n_qubits = len(model.flattened_qubits)
n_electrons = [2, 2]


cou_sys_eig_energies, cou_sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.coulomb_model.fock_hamiltonian,
        n_qubits=len(model.flattened_qubits),
    ),
    particle_number=n_electrons,
    expanded=True,
)
sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
    sparse_operator=get_sparse_operator(
        model.fock_hamiltonian,
        n_qubits=len(model.flattened_qubits),
    ),
    particle_number=n_electrons,
    expanded=True,
)


eig_fids = state_fidelity_to_eigenstates(sys_eig_states[:, 0], cou_sys_eig_states)
print("fids")
for fid, ind in zip(eig_fids, range(cou_sys_eig_states.shape[1])):
    energy = model.hamiltonian.expectation_from_state_vector(
        cou_sys_eig_states[:, ind],
        qubit_map={k: n for n, k in enumerate(model.flattened_qubits)},
    )
    print(f"|E_{ind:0}>:fid: {np.abs(fid):.4f} E: {energy:.3f}")
print(f"sum fids {sum(eig_fids)}")
