from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from building_blocks import (
    get_cheat_sweep,
    get_cheat_coupler,
    get_Z_env,
    get_cheat_coupler_list,
)

from utils import (
    expectation_wrapper,
    ketbra,
    state_fidelity_to_eigenstates,
)
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    spin_dicke_state,
    trace_norm,
)
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
import matplotlib.pyplot as plt


def get_cheat_couplers_of_model(
    model: FermiHubbardModel, n_electrons: list, noise: float = 0
):
    sys_qubits = model.flattened_qubits

    _, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    n_env_qubits = 1
    env_qubits, _, _, _, env_eig_states = get_Z_env(n_qubits=n_env_qubits)

    couplers = get_cheat_coupler_list(
        sys_eig_states=sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0,),
        noise=noise,
    )

    return couplers


nx = 2
ny = 2
n_electrons = [2, 1]
tunneling = 1
coulomb = 2
qid_shape = 9 * (2,)

model = FermiHubbardModel(
    x_dimension=nx, y_dimension=ny, tunneling=tunneling, coulomb=coulomb
)

free_model = model.non_interacting_model

couplers = get_cheat_couplers_of_model(model=model, n_electrons=n_electrons)
free_couplers = get_cheat_couplers_of_model(model=free_model, n_electrons=n_electrons)

for ind in range(len(couplers)):
    fid = trace_norm(couplers[ind], free_couplers[ind])
    print(f"coupler {ind} fidelity: {fid}")
