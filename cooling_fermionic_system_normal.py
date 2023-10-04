import sys

# tsk tsk
sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")


import cirq
import numpy as np
from coolerClass import Cooler, get_total_spectra_at_given_omega
from cooling_building_blocks import (
    get_moving_ZY_coupler_list,
    get_moving_ZYZY_coupler_list,
    get_Z_env,
    get_ZY_coupler,
    get_moving_paulipauli_coupler_list,
    get_moving_fsim_coupler_list,
    get_hamiltonian_coupler,
)
from cooling_utils import expectation_wrapper
from openfermion import get_sparse_operator, jw_hartree_fock_state

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import (
    is_subspace_gs_global,
    jw_eigenspectrum_at_particle_number,
    spin_dicke_mixed_state,
    spin_dicke_state,
)


def pick_ground_state(name: str, n_sys_qubits: int, n_electrons: list):
    if name == "hartree_fock":
        return jw_hartree_fock_state(
            n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
        )
    elif name == "dicke":
        return spin_dicke_state(
            n_qubits=n_sys_qubits, n_electrons=n_electrons, right_to_left=True
        )
    elif name == "mixed":
        return spin_dicke_mixed_state(
            n_qubits=n_sys_qubits, n_electrons=n_electrons, right_to_left=True
        )


def __main__(args):
    # model stuff
    model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [2, 1]
    is_subspace_gs_global(model, n_electrons)
    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)

    sys_initial_state = pick_ground_state("hartree_fock", n_sys_qubits, n_electrons)
    sys_eigenspectrum, sys_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_state = sys_eigenstates[:, np.argmin(sys_eigenspectrum)]
    sys_ground_energy = np.min(sys_eigenspectrum)

    sys_initial_energy = expectation_wrapper(
        model.hamiltonian, sys_initial_state, model.flattened_qubits
    )
    sys_ground_energy_exp = expectation_wrapper(
        model.hamiltonian, sys_ground_state, model.flattened_qubits
    )

    fidelity = cirq.fidelity(
        sys_initial_state,
        sys_ground_state,
        qid_shape=(2,) * (len(model.flattened_qubits)),
    )
    print("initial fidelity: {}".format(fidelity))
    print("ground energy from spectrum: {}".format(sys_ground_energy))
    print("ground energy from model: {}".format(sys_ground_energy_exp))
    print("initial energy from model: {}".format(sys_initial_energy))

    # n_env_qubits = n_sys_qubits
    n_env_qubits = 1

    env_qubits, env_ground_state, env_ham, env_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    # coupler
    # coupler_list = get_hamiltonian_coupler(model.hamiltonian, env_qubits=env_qubits)
    coupler_list = get_moving_ZY_coupler_list(sys_qubits, env_qubits)
    # coupler_list = [
    #     get_moving_fsim_coupler_list(sys_qubits, env_qubits),
    #     get_moving_ZY_coupler_list(sys_qubits, env_qubits),
    # ]
    # coupler_list = get_moving_ZY_coupler_list(sys_qubits, env_qubits)
    # get environment ham sweep values
    spectrum_width = max(sys_eigenspectrum) - min(sys_eigenspectrum)

    min_gap = sorted(np.abs(np.diff(sys_eigenspectrum)))[0]

    # call cool

    cooler = Cooler(
        sys_hamiltonian=model.hamiltonian,
        n_electrons=n_electrons,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=coupler_list,
        verbosity=7,
    )

    n_rep = 1
    ansatz_options = {"beta": 1e-4, "mu": 0.1, "c": 1e-5}
    weaken_coupling = 100

    start_omega = 1.1 * spectrum_width
    stop_omega = 0.1 * min_gap

    print("start: {:.4f} stop: {:.4f}".format(start_omega, stop_omega))

    fidelities, sys_energies, omegas, env_energies = cooler.big_brain_cool(
        start_omega=start_omega,
        stop_omega=stop_omega,
        ansatz_options=ansatz_options,
        n_rep=n_rep,
        weaken_coupling=weaken_coupling,
        coupler_indexing=True,
    )

    print(sys_eigenspectrum)

    supp_eigenspectra = []
    for omega in sys_eigenspectrum:
        subspace_eigvals, subspace_eigvecs = get_total_spectra_at_given_omega(
            cooler=cooler,
            n_electrons=n_electrons,
            omega=omega,
            weaken_coupling=weaken_coupling,
        )
        supp_eigenspectra.append(subspace_eigvals)
    print("Final Fidelity: {}".format(fidelities[-1][-1]))

    cooler.plot_controlled_cooling(
        fidelities=fidelities,
        sys_energies=sys_energies,
        env_energies=env_energies,
        omegas=omegas,
        eigenspectrums=[
            sys_eigenspectrum,
        ],
    )
    print(sys_energies[0][0] - sys_energies[0][-1])
    print(np.sum(env_energies[0]))


if __name__ == "__main__":
    __main__(sys.argv)
