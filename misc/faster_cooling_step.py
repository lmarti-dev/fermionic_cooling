from utils import time_evolve_density_matrix, spin_dicke_state, ketbra
from building_blocks import get_Z_env, get_cheat_coupler
import time
import numpy as np
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from cirq import X, Y
import cupy as cp


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
    model = FermiHubbardModel(x_dimension=x, y_dimension=y, tunneling=1, coulomb=2)
    n_electrons = [2, 2]
    n_sys_qubits = len(model.flattened_qubits)
    env_qubits, env_ground_state, env_ham, env_eig_energies, env_eig_states = get_Z_env(
        n_qubits=n
    )
    sys_dicke = spin_dicke_state(
        n_qubits=n_sys_qubits, Nf=n_electrons, right_to_left=False
    )
    coupler = prod(
        *(np.random.choice((X, Y))(q1) for q1 in model.flattened_qubits)
    ) * prod(*(np.random.choice((X, Y))(q2) for q2 in env_qubits))
    coupler_mat = which.array(
        coupler.matrix(qubits=model.flattened_qubits + env_qubits), dtype=complex
    )
    sys_mat = which.array(
        model.hamiltonian.matrix(qubits=model.flattened_qubits), dtype=complex
    )
    env_mat = which.array(env_ham.matrix(qubits=env_qubits), dtype=complex)
    ham = which.kron(sys_mat, env_mat) + coupler_mat
    rho = which.kron(
        which.array(ketbra(sys_dicke), dtype=complex),
        which.array(ketbra(env_ground_state), dtype=complex),
    )
    return ham, rho


methods = ("diag", "expm")
n_steps = 100

times = {m: {} for m in methods}

results = {k: None for k in methods}
shapes = ((2, 2, 1), (2, 2, 2), (2, 2, 3))
shapes = ((2, 2, 1),)

for ind, (x, y, n) in enumerate(shapes):
    for method in methods:
        if method == "diag":
            which = cp
        else:
            which = np
        start = time.time()
        for _ in range(n_steps):
            # inside to mimick changing ham
            ham, rho = get_ham_rho(x, y, n, which=which)
            t = 1000 * np.random.rand()
            Ut_rho_Utd = time_evolve_density_matrix(
                ham=ham, rho=rho, t=t, method=method
            )
        end = time.time()
        elapsed = end - start

        times[method][str((x, y, n))] = elapsed
        results[method] = Ut_rho_Utd
        print(method, (x, y, n), elapsed)
    print(f"{method}: {np.all(np.isclose(results[method],results['expm']))}")
