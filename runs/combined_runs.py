import matplotlib.pyplot as plt
import numpy as np
from adiabatic_sweep import (
    fermion_to_dense,
    get_sweep_hamiltonian,
    run_sweep,
    get_instantaneous_ground_states,
)
from adiabaticCooler import AdiabaticCooler
from building_blocks import (
    control_function,
    get_cheat_coupler_list,
    get_cheat_sweep,
    get_Z_env,
)
from cirq import PauliSum, Qid
from coolerClass import Cooler
from openfermion import get_sparse_operator, jordan_wigner, jw_hartree_fock_state
from utils import (
    get_extrapolated_superposition,
    get_min_gap,
    get_slater_spectrum,
    get_onsite_spectrum,
    ketbra,
    trace_out_env,
)

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
    wrapping_slice,
)

# there are five cases
# 1. sweep only (done)
# 2. cooling only (done)
# 3. cooling then sweep
# 4. sweep then cooling
# 5. sweep and cooling


def get_couplers_for_sweep(
    model: FermiHubbardModel,
    n_steps: int,
    which: str,
    env_up: np.ndarray,
    n_electrons: list,
    energy_approx_lvl: int = 1,
):
    omegas = np.zeros((n_steps,))
    couplers = []

    params = model.to_json_dict()["constructor_params"]
    tunneling = params["tunneling"]
    coulomb = params["coulomb"]

    if which == "coulomb":
        cou_vals = np.linspace(0, coulomb, n_steps)
        tun_vals = (tunneling,) * len(cou_vals)
    elif which == "tunneling":
        tun_vals = np.linspace(0, tunneling, n_steps)
        cou_vals = (coulomb,) * len(tun_vals)

    for ind, (tun, cou) in enumerate(zip(tun_vals, cou_vals)):
        model = FermiHubbardModel(
            x_dimension=params["x_dimension"],
            y_dimension=params["y_dimension"],
            tunneling=tun,
            coulomb=cou,
        )
        m_eig_energies, m_eig_states = jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(model.fock_hamiltonian),
            particle_number=n_electrons,
            expanded=True,
        )
        unique_energies = np.array(list(sorted(set(m_eig_energies))))
        unique_diffs = np.diff(unique_energies)
        omegas[ind] = np.mean(np.abs(unique_diffs[:energy_approx_lvl]))
        coupler = np.kron(
            np.outer(m_eig_states[:, 0], np.conjugate(m_eig_states[:, 0])),
            env_up,
        )
        coupler = coupler + np.conjugate(np.transpose(coupler))
        couplers.append(coupler)

    return couplers, omegas


def adiabatic_cooling(which="coulomb"):
    tunneling = 1
    coulomb = 2

    x_dimension = 2
    y_dimension = 2

    n_electrons = [2, 2]

    if which == "coulomb":
        close_model_tunneling = tunneling
        close_model_coulomb = 1e-5
        start_model_tunneling = tunneling
        start_model_coulomb = 0

    elif which == "tunneling":
        close_model_tunneling = 1e-5
        close_model_coulomb = coulomb
        start_model_tunneling = 0
        start_model_coulomb = coulomb

    model = FermiHubbardModel(
        x_dimension=2, y_dimension=2, tunneling=tunneling, coulomb=coulomb
    )
    close_model = FermiHubbardModel(
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        tunneling=close_model_tunneling,
        coulomb=close_model_coulomb,
    )
    close_model = FermiHubbardModel(
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        tunneling=close_model_tunneling,
        coulomb=close_model_coulomb,
    )
    start_model = FermiHubbardModel(
        x_dimension=x_dimension,
        y_dimension=y_dimension,
        tunneling=start_model_tunneling,
        coulomb=start_model_coulomb,
    )

    n_env_qubits = 1
    n_steps = 1000
    weaken_coupling = 5

    sys_qubits = model.flattened_qubits
    n_sys_qubits = len(model.flattened_qubits)

    ham_start = start_model.hamiltonian
    ham_stop = model.hamiltonian

    if which == "coulomb":
        start_eig_energies, start_eig_states = get_slater_spectrum(
            model, n_electrons=n_electrons
        )
    elif which == "tunneling":
        start_eig_energies, start_eig_states = get_onsite_spectrum(
            model, n_electrons=n_electrons
        )
    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )
    close_eig_energies, close_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(close_model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    spectrum_width = np.abs(np.max(sys_eig_energies) - np.min(sys_eig_energies))
    total_time = (
        10 * spectrum_width / (get_min_gap(sys_eig_energies, threshold=1e-12) ** 2)
    )

    sys_ground_state = sys_eig_states[:, 0]
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )

    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))

    couplers, omegas = get_couplers_for_sweep(
        model=model,
        n_steps=n_steps,
        which=which,
        env_up=env_up,
        n_electrons=n_electrons,
        energy_approx_lvl=3,
    )

    spectrum = sys_eig_energies
    sys_hartree_fock = jw_hartree_fock_state(
        n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons)
    )

    slater_superpos = np.sum(start_eig_states[:, 1:3], axis=1)
    slater_superpos /= np.linalg.norm(slater_superpos)
    sys_initial_state = ketbra(start_eig_states[:, 0])

    adiabatic_cooler = AdiabaticCooler(
        sys_hamiltonian=model.hamiltonian,
        n_electrons=n_electrons,
        sys_qubits=model.flattened_qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
        ham_start=ham_start,
        ham_stop=ham_stop,
    )

    alphas = omegas / (weaken_coupling * len(adiabatic_cooler.sys_qubits))
    # evolution_times = 2.5 * np.pi / np.abs(alphas)
    evolution_times = (total_time / n_steps,) * n_steps
    (
        sweep_fidelities,
        _,
        _,
    ) = adiabatic_cooler.adiabatic_sweep(total_time=total_time, n_steps=n_steps)
    (
        sys_fidelities,
        sys_energies,
        env_energies,
        total_density_matrix,
    ) = adiabatic_cooler.adiabatic_cool(
        evolution_times=evolution_times,
        alphas=alphas,
        omegas=omegas,
    )

    fig, ax = plt.subplots()
    x = np.linspace(0, total_time, n_steps + 1)

    ax.plot(x, sweep_fidelities, "r", label="Sweep")
    ax.plot(x, sys_fidelities, "b", label="Cool + sweep")

    ax.set_xlabel("Time")
    ax.set_ylabel("Fidelity")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    adiabatic_cooling("tunneling")
