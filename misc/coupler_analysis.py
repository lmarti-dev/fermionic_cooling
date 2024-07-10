import sys

# tsk tsk
sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/qutlet/")


import numpy as np
from building_blocks import (
    get_moving_fsim_coupler_list,
    get_moving_ZY_coupler_list,
    get_Z_env,
)
from utils import (
    coupler_fidelity_to_ground_state_projectors,
)
from openfermion import get_sparse_operator

from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from qutlet.utilities import (
    jw_eigenspectrum_at_particle_number,
)

import matplotlib.pyplot as plt
import re


def process_pauli_sum_str(psum_str: str):
    psum_str = re.sub(r"\*", "", psum_str)
    psum_str = re.sub(r"[0-9]+\.[0-9]+", "", psum_str)
    psum_str = re.sub(r"\(q(.*?)\)", r"\1", psum_str)
    psum_str = re.sub(r" ", "", psum_str)
    return psum_str


if __name__ == "__main__":
    n_electrons = [2, 1]
    model = FermiHubbardModel(
        lattice_dimensions=(2, 2), n_electrons=n_electrons, tunneling=1, coulomb=2
    )

    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)

    sys_eigenspectrum, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_state = sys_eig_states[:, np.argmin(sys_eigenspectrum)]
    sys_ground_energy = np.min(sys_eigenspectrum)

    n_env_qubits = 1

    env_qubits, env_ground_state, env_ham, env_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    fix, axes = plt.subplots(nrows=2)

    coupler_list_fsim = get_moving_fsim_coupler_list(sys_qubits, env_qubits)
    coupler_list_ZY = get_moving_ZY_coupler_list(sys_qubits, env_qubits)
    for list_n, coupler_list in enumerate([coupler_list_fsim, coupler_list_ZY]):
        couplers = []
        measures = []
        for coupler in coupler_list:
            couplers.append(coupler)
            print(coupler)
            measures.append(
                coupler_fidelity_to_ground_state_projectors(
                    coupler=coupler.matrix(qubits=sys_qubits + env_qubits),
                    sys_eig_states=sys_eig_states,
                    env_eig_states=env_eig_states,
                    exponentiate=True,
                )
            )
        for coupler, measure in zip(couplers, measures):
            axes[list_n].plot(
                range(1, sys_eig_states.shape[1]),
                np.abs(measure),
                label=process_pauli_sum_str(str(coupler)),
            )

            axes[list_n].legend()
    axes[-1].set_xlabel(r"$k: |\psi_0\rangle \langle \psi_k|$")
    plt.show()
