# define two Pauli Sums which are H_S (FermionicModel) and H_E (sum of Z simplest, but perhaps better choice like Heisenberg)
# and third PauliSum interaction V
# do time evolution H_S + beta * H_E + alpha + V
# 1) trace out E from the density matrix of the whole thing
# rentensor this traced out mixed state with gs of H_E
# 2) *not implemented* you could either measure E and restart if it's still in gs
# or apply gs circuit if it's not

# 1) measure energy, fidelity of density matrices with correct gs
# after every step

# since the population is distributed across all excited energy levels,
# we need to sweep the coupling in H_E and change V (V is |E_j><E_i| but we dont know it we just approximate it)
# so that it matches the cooling trnasition that we want to simulate
# it's good to know the eigenspectrum of the system (but irl we don't know)
# get max(eigenvalues)-min(eigenvalues)
# logsweep is log spaced omega sampling

import numpy as np
import cirq
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply, eigsh, expm
from typing import Iterable
import time
import matplotlib.pyplot as plt
from openfermion import FermionOperator
import itertools
import multiprocessing as mp


class Cooler:
    def __init__(
        self,
        sys_hamiltonian: cirq.PauliSum,
        sys_qubits: Iterable[cirq.Qid],
        sys_initial_state: np.ndarray,
        sys_ground_state: np.ndarray,
        env_hamiltonian: cirq.PauliSum,
        env_qubits: Iterable[cirq.Qid],
        env_ground_state: np.ndarray,
        sys_env_coupling: cirq.PauliSum,
        verbosity: int = 0,
    ):
        self.verbosity = verbosity
        self.sys_hamiltonian = sys_hamiltonian
        self.sys_qubits = sys_qubits
        self.sys_initial_state = sys_initial_state
        self.sys_ground_state = sys_ground_state
        self.env_hamiltonian = env_hamiltonian
        self.env_qubits = env_qubits
        self.env_ground_state = env_ground_state
        self.sys_ground_energy = np.real(
            sys_hamiltonian.expectation_from_state_vector(
                sys_ground_state,
                qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
            )
        )
        self.sys_env_coupling = sys_env_coupling

    def verbose_print(self, s: str, message_level: int = 1):
        if int(self.verbosity) >= message_level:
            print(s)

    def cooling_hamiltonian(self, env_coupling: float, alpha: float):
        if isinstance(self.sys_env_coupling, cirq.PauliSum):
            coupler = self.sys_env_coupling.matrix(qubits=self.total_qubits)
        else:
            coupler = self.sys_env_coupling
        return (
            self.sys_hamiltonian.matrix(qubits=self.total_qubits)
            + env_coupling * self.env_hamiltonian.matrix(qubits=self.total_qubits)
            + float(alpha) * coupler
        )

    @property
    def total_qubits(self):
        return self.sys_qubits + self.env_qubits

    @property
    def total_initial_state(self):
        if is_density_matrix(self.sys_initial_state):
            return np.kron(self.sys_initial_state, ketbra(self.env_ground_state))
        else:
            return ketbra(np.kron(self.sys_initial_state, self.env_ground_state))

    @property
    def env_ground_density_matrix(self):
        return ketbra(self.env_ground_state)

    def sys_fidelity(self, state: np.ndarray):
        return cirq.fidelity(
            state.astype("complex_"),
            self.sys_ground_state,
            qid_shape=(2,) * (len(self.sys_qubits)),
        )

    def sys_energy(self, sys_state: np.ndarray):
        return expectation_wrapper(self.sys_hamiltonian, sys_state, self.sys_qubits)

    def cool(
        self,
        evolution_times: np.ndarray,
        alphas: np.ndarray,
        sweep_values: Iterable[float],
    ):
        initial_density_matrix = self.total_initial_state
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        energies = []

        fidelity = self.sys_fidelity(self.sys_initial_state)
        energy = self.sys_energy(self.sys_initial_state)

        self.verbose_print(
            "initial fidelity to gs: {}, initial energy of traced out rho: {}, ground energy: {}".format(
                fidelity, energy, self.sys_ground_energy
            )
        )

        fidelities.append(fidelity)
        energies.append(energy)

        for step, env_coupling in enumerate(sweep_values):
            # print("=== step: {}/{} ===".format(step, len(sweep_values)))
            fidelity, energy, total_density_matrix = self.cooling_step(
                total_density_matrix=total_density_matrix,
                env_coupling=env_coupling,
                alpha=alphas[step],
                evolution_time=evolution_times[step],
            )
            fidelities.append(fidelity)
            energies.append(energy)
            self.verbose_print(has_increased(fidelity, fidelities[-2], "fidelity"))
            self.verbose_print(has_increased(energy, energies[-2], "energy"))
            self.verbose_print(
                "fidelity to gs: {}, energy diff of traced out rho: {}".format(
                    fidelity, energy - self.sys_ground_energy
                )
            )
        return fidelities, energies

    def cooling_step(
        self,
        total_density_matrix: np.ndarray,
        alpha: float,
        env_coupling: float,
        evolution_time: float,
    ):
        cooling_hamiltonian = self.cooling_hamiltonian(env_coupling, alpha)

        self.verbose_print("env coupling value: {}".format(env_coupling))
        self.verbose_print("alpha value: {}".format(alpha))
        self.verbose_print("evolution_time value: {}".format(evolution_time))

        self.verbose_print("evolving...")
        total_density_matrix = time_evolve_density_matrix(
            ham=cooling_hamiltonian,  # .matrix(qubits=self.total_qubits),
            rho=total_density_matrix,
            t=evolution_time,
            method="expm",
        )
        traced_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )
        self.verbose_print("computing values...")
        fidelity = self.sys_fidelity(traced_density_matrix)
        energy = self.sys_energy(traced_density_matrix)

        self.verbose_print("retensoring...")
        total_density_matrix = np.kron(
            traced_density_matrix, self.env_ground_density_matrix
        )
        return fidelity, energy, total_density_matrix

    def forced_cool(
        self,
        evolution_times: np.ndarray,
        alphas: np.ndarray,
        sweep_values: Iterable[float],
    ):
        initial_density_matrix = ketbra(self.total_initial_state)
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        energies = []

        fidelity = self.sys_fidelity(self.sys_initial_state)
        energy = self.sys_energy(self.sys_initial_state)

        self.verbose_print(
            "initial fidelity to gs: {}, initial energy of traced out rho: {}, ground energy: {}".format(
                fidelity, energy, self.sys_ground_energy
            )
        )

        fidelities.append(fidelity)
        energies.append(energy)

        for step, env_coupling in enumerate(sweep_values):
            fidelity, energy, total_density_matrix = self.cooling_step(
                total_density_matrix=total_density_matrix,
                env_coupling=env_coupling,
                alpha=alphas[step],
                evolution_time=evolution_times[step],
            )
            fidelities.append(fidelity)
            energies.append(energy)
            self.verbose_print(
                "fidelity to gs: {}, energy diff of traced out rho: {}".format(
                    fidelity, energy - self.sys_ground_energy
                )
            )

            iteration_number = 1
            while energy + 1e-4 < energies[-2]:
                self.verbose_print("while loop iteration {}".format(iteration_number))
                fidelity, energy, total_density_matrix = self.cooling_step(
                    total_density_matrix=total_density_matrix,
                    env_coupling=env_coupling,
                    alpha=alphas[step],
                    evolution_time=evolution_times[step],
                )
                fidelities.append(fidelity)
                energies.append(energy)
                self.verbose_print(
                    has_increased(fidelity, fidelities[-2], "while fidelity")
                )
                self.verbose_print(has_increased(energy, energies[-2], "while energy"))
                self.verbose_print(
                    "fidelity to gs: {}, energy diff of traced out rho: {}".format(
                        fidelity, energy - self.sys_ground_energy
                    )
                )
                iteration_number += 1
        return fidelities, energies

    def plot_cooling(
        self, energies: list, fidelities: list, sys_eigenspectrum: np.ndarray = None
    ):
        if sys_eigenspectrum is None:
            nrows = 2
        else:
            nrows = 3
        fig, axes = plt.subplots(nrows=nrows, figsize=(5, 3))
        plt.rcParams.update({"font.size": 22})

        axes[0].plot(
            range(len(fidelities)),
            fidelities,
        )
        axes[0].set_ylabel(r"$\langle \psi_{cool} | \psi_{gs} \rangle$", labelpad=0)
        axes[1].plot(
            range(len(energies)),
            (np.array(energies) - self.sys_ground_energy)
            / np.abs(self.sys_ground_energy),
        )
        axes[1].set_ylabel(r"$\frac{E_{cool}-E_0}{|E_0|}$", labelpad=0)
        if sys_eigenspectrum is not None:
            axes[2].hlines(sys_eigenspectrum, xmin=-2, xmax=2)
            axes[2].set_ylabel("Eigenenergies")

        plt.show()


def mean_gap(spectrum: np.ndarray):
    return float(np.mean(np.diff(spectrum)))


def get_cheat_coupler(sys_eig_states, env_eig_states, qubits, to_psum: bool = False):
    coupler = 0
    env_up = np.outer(env_eig_states[:, 1], np.conjugate(env_eig_states[:, 0]))
    for k in range(1, sys_eig_states.shape[1]):
        coupler += np.kron(
            np.outer(sys_eig_states[:, 0], np.conjugate(sys_eig_states[:, k])),
            env_up,
        )
    if to_psum:
        return ndarray_to_psum(
            coupler + np.conjugate(np.transpose(coupler)), qubits=qubits
        )
    else:
        return coupler + np.conjugate(np.transpose(coupler))


def get_log_sweep(spectrum_width: np.ndarray, n_steps: int):
    return spectrum_width * (np.logspace(start=0, stop=-5, base=10, num=n_steps))


def get_cheat_sweep(spectrum: np.ndarray, n_steps: int = None):
    res = []
    if n_steps is None:
        n_rep = 1
    else:
        n_rep = int(n_steps / (len(spectrum) - 1))
    for k in range(len(spectrum) - 1, 0, -1):
        res.append(spectrum[k] - spectrum[0])
    return np.tile(np.array(res), n_rep)


def get_lin_sweep(spectrum: np.ndarray, n_steps: int):
    min_gap = sorted(np.abs(np.diff(spectrum)))[0]
    spectrum_width = max(spectrum) - min(spectrum)
    return np.linspace(start=spectrum_width, stop=min_gap, num=n_steps)


def is_density_matrix(state):
    return len(state.shape) == 2


def expectation_wrapper(observable, state, qubits):
    if is_density_matrix(state):
        return np.real(
            observable.expectation_from_density_matrix(
                state.astype("complex_"),
                qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
            )
        )
    else:
        return np.real(
            observable.expectation_from_state_vector(
                state.astype("complex_"),
                qubit_map={k: v for k, v in zip(qubits, range(len(qubits)))},
            )
        )


def get_psum_qubits(psum: cirq.PauliSum) -> Iterable[cirq.Qid]:
    qubits = []
    for pstr in psum:
        qubits.extend(pstr.keys())
    return tuple(set(qubits))


def dagger(U: np.ndarray) -> np.ndarray:
    return np.transpose(np.conj(U))


def time_evolve_state(ham: np.ndarray, ket: np.ndarray, t: float):
    return expm_multiply(A=-1j * t * ham, B=ket)


def time_evolve_density_matrix(
    ham: np.ndarray, rho: np.ndarray, t: float, method: str = "expm_multiply"
):
    # print("timing...")
    start = time.time()
    if method == "expm_multiply":
        # can be extremely slow
        Ut_rho = expm_multiply(A=-1j * t * ham, B=rho)
        Ut_rho_Utd = dagger(expm_multiply(A=-1j * t * ham, B=dagger(Ut_rho)))
    elif method == "expm":
        Ut = expm(-1j * t * ham)
        Ut_rho_Utd = Ut @ rho @ Ut.transpose().conjugate()
    end = time.time()
    # print("time evolution took: {} sec".format(end - start))
    if not cirq.is_hermitian(Ut_rho_Utd):
        raise ValueError("time-evolved density matrix is not hermitian")
    return Ut_rho_Utd


def get_ground_state(ham: cirq.PauliSum, qubits: Iterable[cirq.Qid]) -> np.ndarray:
    _, ground_state = eigsh(ham.matrix(qubits=qubits), k=1, which="SA")
    return ground_state


def trace_out_env(
    rho: np.ndarray,
    n_sys_qubits: int,
    n_env_qubits: int,
):
    # reshaped_rho = np.reshape(rho, (2,) * 2 * (n_sys_qubits + n_env_qubits))
    # traced_rho = cirq.partial_trace(
    #     tensor=reshaped_rho, keep_indices=range(n_sys_qubits)
    # )
    # reshaped_traced_rho = np.reshape(traced_rho, (2**n_sys_qubits, 2**n_sys_qubits))
    # print("tracing out environment...")

    traced_rho = np.zeros((2**n_sys_qubits, 2**n_sys_qubits), dtype="complex_")
    # print("traced rho shape: {} rho shape: {}".format(traced_rho.shape, rho.shape))
    for iii in range(2**n_sys_qubits):
        for jjj in range(2**n_sys_qubits):
            # take rho[i*env qubits:i*env qubits + env qubtis]
            traced_rho[iii, jjj] = np.trace(
                rho[
                    iii * (2**n_env_qubits) : (iii + 1) * (2**n_env_qubits),
                    jjj * (2**n_env_qubits) : (jjj + 1) * (2**n_env_qubits),
                ]
            )
    return traced_rho


def ketbra(ket: np.ndarray):
    return np.outer(ket, dagger(ket))


def has_increased(val_current: float, val_previous: float, quantname: str):
    return "{q} has {i}".format(
        q=quantname,
        i="increased" if np.real(val_current) > np.real(val_previous) else "decreased",
    )


def fermionic_spin_and_number(n_qubits):
    n_up_op = sum(
        [FermionOperator("{x}^ {x}".format(x=x)) for x in range(0, n_qubits, 2)]
    )
    n_down_op = sum(
        [FermionOperator("{x}^ {x}".format(x=x)) for x in range(1, n_qubits, 2)]
    )
    n_total_op = sum(n_up_op, n_down_op)
    return n_up_op, n_down_op, n_total_op


def pauli_string_coeff_dispatcher(data):
    return pauli_string_coeff(*data)


def pauli_string_coeff(
    mat: np.ndarray, pauli_product: list[cirq.Pauli], qubits: list[cirq.Qid]
):
    pauli_string = cirq.PauliString(*[m(q) for m, q in zip(pauli_product, qubits)])
    pauli_matrix = pauli_string.matrix(qubits=qubits)
    coeff = np.trace(mat @ pauli_matrix) / mat.shape[0]
    return coeff, pauli_string


def ndarray_to_psum(
    mat: np.ndarray,
    qubits: list[cirq.Qid] = None,
    n_jobs: int = 32,
    verbose: bool = False,
) -> cirq.PauliSum:
    if len(list(set(mat.shape))) != 1:
        raise ValueError("the matrix is not square")
    n_qubits = int(np.log2(mat.shape[0]))
    if qubits is None:
        qubits = cirq.LineQubit.range(n_qubits)
    pauli_matrices = (cirq.I, cirq.X, cirq.Y, cirq.Z)
    pauli_products = itertools.product(pauli_matrices, repeat=n_qubits)
    pauli_sum = cirq.PauliSum()
    if n_jobs > 1:
        pool = mp.Pool(n_jobs)
        results = pool.starmap(
            pauli_string_coeff,
            ((mat, pauli_product, qubits) for pauli_product in pauli_products),
        )

        for result in results:
            coeff, pauli_string = result
            pauli_sum += cirq.PauliString(pauli_string, coeff)
        pool.close()
        pool.join()
    else:
        for pauli_product in pauli_products:
            if verbose:
                print(pauli_product, coeff)
            if not np.isclose(np.abs(coeff), 0):
                pauli_sum += cirq.PauliString(pauli_string, coeff)
    return pauli_sum
