from utils import time_evolve_density_matrix, spin_dicke_state, ketbra
from building_blocks import get_Z_env, get_cheat_couplers
import time
import numpy as np
from qutlet.models.fermi_hubbard_model import FermiHubbardModel
from qutlet.utilities import (
    jw_eigenspectrum_at_particle_number,
    jw_get_true_ground_state_at_particle_number,
)
from cirq import X, Y
from cooler_class import Cooler
from openfermion import get_sparse_operator, jw_hartree_fock_state


def setup_cooler(x, y, n_env_qubits, method):
    n_electrons = [2, 2]
    model = FermiHubbardModel(
        lattice_dimensions=(x, y), n_electrons=n_electrons, tunneling=1, coulomb=2
    )
    sys_qubits = model.qubits
    n_sys_qubits = len(sys_qubits)

    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n_env_qubits
    )
    sparse_op = get_sparse_operator(
        model.non_interacting_model.fock_hamiltonian,
        n_qubits=len(model.qubits),
    )
    free_sys_eig_energies, free_sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=sparse_op,
        particle_number=n_electrons,
        expanded=True,
    )
    sys_ground_energy, sys_ground_state = jw_get_true_ground_state_at_particle_number(
        sparse_operator=sparse_op, particle_number=n_electrons
    )
    sys_initial_state = ketbra(
        jw_hartree_fock_state(n_orbitals=n_sys_qubits, n_electrons=sum(n_electrons))
    )
    couplers = get_cheat_couplers(
        sys_eig_states=free_sys_eig_states,
        env_eig_states=env_eig_states,
        qubits=sys_qubits + env_qubits,
        gs_indices=(0, 1, 2),
        noise=0,
    )  # Interaction only on Qubit 0?

    cooler = Cooler(
        sys_hamiltonian=model.hamiltonian,
        n_electrons=n_electrons,
        sys_qubits=model.qubits,
        sys_ground_state=sys_ground_state,
        sys_initial_state=sys_initial_state,
        env_hamiltonian=env_ham,
        env_qubits=env_qubits,
        env_ground_state=env_ground_state,
        sys_env_coupler_data=couplers,
        verbosity=5,
        time_evolve_method=method,
    )
    return cooler


def prod(*args):
    x = args[0]
    for ii in range(1, len(args)):
        x *= args[ii]
    return x


def time_func(func: callable):
    start = time.time()
    func()
    end = time.time()
    return (end - start,)


def get_ham_rho(x, y, n, which=np):
    n_electrons = [2, 2]
    model = FermiHubbardModel(
        lattice_dimensions=(x, y), n_electrons=n_electrons, tunneling=1, coulomb=2
    )
    n_sys_qubits = len(model.qubits)
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n
    )
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, n_electrons=n_electrons, right_to_left=False
    )
    coupler = prod(*(np.random.choice((X, Y))(q1) for q1 in model.qubits)) * prod(
        *(np.random.choice((X, Y))(q2) for q2 in env_qubits)
    )
    coupler_mat = which.array(
        coupler.matrix(qubits=model.qubits + env_qubits), dtype=complex
    )
    sys_mat = which.array(model.hamiltonian.matrix(qubits=model.qubits), dtype=complex)
    env_mat = which.array(env_ham.matrix(qubits=env_qubits), dtype=complex)
    ham = which.kron(sys_mat, env_mat) + coupler_mat
    rho = which.kron(
        which.array(ketbra(sys_dicke), dtype=complex),
        which.array(ketbra(env_ground_state), dtype=complex),
    )
    return ham, rho


def time_evolve_step(x, y, n, which):
    ham, rho = get_ham_rho(x, y, n, which=which)
    t = 1000 * np.random.rand()
    Ut_rho_Utd = time_evolve_density_matrix(ham=ham, rho=rho, t=t, method=method)
    return Ut_rho_Utd


def full_cooling_step(cooler: Cooler, which=np):
    alpha = which.random.rand() + 0.1
    env_coupling = which.random.rand() * 10
    evolution_time = 2.5 * np.pi / alpha
    sys_fidelity, sys_energy, env_energy, total_density_matrix = cooler.cooling_step(
        cooler.total_initial_state, alpha, env_coupling, evolution_time
    )
    return total_density_matrix


methods = ("diag", "expm")
n_steps = 30

times = {m: {} for m in methods}

results = {k: None for k in methods}
shapes = ((2, 2, 1), (2, 2, 2), (2, 2, 3))
shapes = ((2, 2, 1), (2, 2, 2), (2, 2, 3))

for ind, (x, y, n) in enumerate(shapes):
    elapsed = []
    for method in methods:
        cooler = setup_cooler(x, y, n_env_qubits=n, method=method)
        for _ in range(n_steps):
            start = time.time()
            # inside to mimick changing ham
            Ut_rho_Utd = full_cooling_step(cooler=cooler)
            end = time.time()
            elapsed.append(end - start)

        times[method][str((x, y, n))] = elapsed
        results[method] = Ut_rho_Utd
        print(
            f"{method} {(x, y, n)} mean: {np.mean(elapsed):.3f} var: {np.var(elapsed):.3f}"
        )

# print(f"{method}: {np.all(np.isclose(results[method],results['expm']))}")
