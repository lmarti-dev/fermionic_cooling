import sys

# tsk tsk
sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")

from fauvqe.models.fermiHubbardModel import FermiHubbardModel

from coolerClass import (
    Cooler,
    expectation_wrapper,
    get_cheat_sweep,
    get_cheat_coupler,
    get_log_sweep,
    get_Z_env,
    ketbra,
)
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
import cirq
from openfermion import get_sparse_operator, jw_hartree_fock_state
import numpy as np
from fauvqe.utilities import spin_dicke_state, qmap


def __main__(args):
    # model stuff
    model = FermiHubbardModel(x_dimension=1, y_dimension=2, tunneling=1, coulomb=2)
    Nf = [1, 1]
    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(sys_qubits)
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(Nf)
    )
    sys_dicke = spin_dicke_state(n_qubits=n_sys_qubits, Nf=Nf, right_to_left=True)
    sys_initial_state = ketbra(sys_hartree_fock)
    sys_eigenspectrum, sys_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(
            model.fock_hamiltonian,
            n_qubits=len(model.flattened_qubits),
        ),
        particle_number=Nf,
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

    env_qubits, env_ground_state, env_ham, env_energies, env_eig_states = get_Z_env(
        n_qubits=n_sys_qubits
    )

    # coupler
    coupler = sum(
        [cirq.Y(sys_qubits[k]) * cirq.Y(env_qubits[k]) for k in range(n_sys_qubits)]
    )  # Interaction only on Qubit 0?
    # coupler = get_cheat_coupler(
    #    sys_eig_states=sys_eigenstates,
    #    env_eig_states=env_eig_states,
    #    qubits=sys_qubits + env_qubits,
    # )  # Interaction only on Qubit 0?
    # coupler = get_cheat_coupler(
    #     sys_eig_states=sys_eigenstates,
    #     env_eig_states=env_eig_states,
    #     qubits=sys_qubits + env_qubits,
    # )
    # coupler = get_cheat_coupler(sys_eigenstates, env_eigenstates)

    # get environment ham sweep values
    spectrum_width = max(sys_eigenspectrum) - min(sys_eigenspectrum)

    min_gap = sorted(np.abs(np.diff(sys_eigenspectrum)))[0]

    # n_steps = 1000
    # sweep_values = get_log_sweep(spectrum_width, n_steps, n_rep)
    # sweep_values = get_cheat_sweep(sys_eigenspectrum, n_steps)
    # np.random.shuffle(sweep_values)
    # coupling strength value
    # alphas = sweep_values / 10 / 4
    # evolution_times = np.pi / (alphas)
    # evolution_time = 1e-3

    # call cool

    cooler = Cooler(
        sys_hamiltonian=model.hamiltonian,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupling=coupler,
        verbosity=5,
    )

    # fidelities, energies = cooler.cool(
    #     alphas=alphas,
    #     evolution_times=evolution_times,
    #     sweep_values=sweep_values,
    # )

    n_rep = 1
    ansatz_options = {"beta": 1e-4, "mu": 0.5, "c": 1e-4}

    fidelities, sys_energies, omegas, env_energies = cooler.big_brain_cool(
        start_omega=1.1 * spectrum_width,
        stop_omega=0.1 * min_gap,
        ansatz_options=ansatz_options,
        n_rep=n_rep,
    )

    print(sys_eigenspectrum)

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

    # cooler.plot_generic_cooling(
    #     energies,
    #     fidelities,
    #     supplementary_data={
    #         "Eigenspectrum": sys_eigenspectrum,
    #         r"$(\frac{\mathrm{d}}{\mathrm{ds}}\omega)^{-2}$": 1
    #         / (np.diff(omegas) ** (-2)),
    #         "Evironement temp.": env_energies,
    #     },
    # )


if __name__ == "__main__":
    __main__(sys.argv)
