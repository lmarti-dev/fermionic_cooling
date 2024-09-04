from qutlet.models.fermi_hubbard_model import FermiHubbardModel

from building_blocks import (
    get_Z_env,
    get_cheat_couplers,
)

from qutlet.utilities import (
    jw_eigenspectrum_at_particle_number,
    trace_norm,
)
from openfermion import get_sparse_operator


def get_cheat_couplers_of_model(
    model: FermiHubbardModel, n_electrons: list, noise: float = 0
):
    sys_qubits = model.qubits

    _, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )

    n_env_qubits = 1
    env_qubits, _, _, _, env_eig_states = get_Z_env(n_qubits=n_env_qubits)

    couplers = get_cheat_couplers(
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
    x_dimension=nx,
    y_dimension=ny,
    n_electrons=n_electrons,
    tunneling=tunneling,
    coulomb=coulomb,
)

free_model = model.non_interacting_model

couplers = get_cheat_couplers_of_model(model=model, n_electrons=n_electrons)
free_couplers = get_cheat_couplers_of_model(model=free_model, n_electrons=n_electrons)

for ind in range(len(couplers)):
    fid = trace_norm(couplers[ind], free_couplers[ind])
    print(f"coupler {ind} fidelity: {fid}")
