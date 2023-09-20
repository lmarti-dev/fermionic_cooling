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

from typing import Iterable

import cirq
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from cooling_utils import (
    expectation_wrapper,
    get_transition_rates,
    has_increased,
    is_density_matrix,
    ketbra,
    time_evolve_density_matrix,
    trace_out_env,
)
from cooling_building_blocks import control_function
from tqdm import tqdm

from fauvqe.utilities import ket_in_subspace


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

        # message for verbose printing
        self.msg = []

    def update_message(self, s: str, message_level: int = 1):
        if int(self.verbosity) >= message_level:
            self.msg.append(s)

    def reset_msg(self):
        self.msg = []

    def print_msg(self):
        print("\n".join(self.msg))
        print("\033[F" * (1 + len(self.msg)))
        self.reset_msg()

    def cooling_hamiltonian(self, env_coupling: float, alpha: float):
        if isinstance(self.sys_env_coupling, (cirq.PauliSum, cirq.PauliString)):
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

    def env_energy(self, env_state: np.ndarray):
        return expectation_wrapper(
            self.env_hamiltonian,
            env_state,
            self.total_qubits,
        )

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

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)

        self.update_message(
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=9,
        )

        fidelities.append(sys_fidelity)
        energies.append(sys_energy)

        for step, env_coupling in tqdm(
            enumerate(sweep_values), total=len(sweep_values)
        ):
            # print("=== step: {}/{} ===".format(step, len(sweep_values)))
            sys_fidelity, sys_energy, _, total_density_matrix = self.cooling_step(
                total_density_matrix=total_density_matrix,
                env_coupling=env_coupling,
                alpha=alphas[step],
                evolution_time=evolution_times[step],
            )
            fidelities.append(sys_fidelity)
            energies.append(sys_energy)
            self.update_message(
                has_increased(sys_fidelity, fidelities[-2], "fidelity"), message_level=9
            )
            self.update_message(
                has_increased(sys_energy, energies[-2], "energy"), message_level=9
            )
            self.update_message(
                "fidelity to gs: {:.4f}, energy diff of traced out rho: {:.4f}".format(
                    sys_fidelity, sys_energy - self.sys_ground_energy
                ),
                message_level=9,
            )
        return fidelities, energies

    def big_brain_cool(
        self,
        start_omega: float,
        stop_omega: float,
        n_rep: int = 1,
        ansatz_options: dict = {},
        coupler_list: list[cirq.PauliSum] = None,
        weaken_coupling: float = 100,
    ):
        initial_density_matrix = self.total_initial_state
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        sys_energies = []
        omegas = []
        env_energies = []

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)
        env_energy = self.env_energy(self.total_initial_state)

        self.update_message(
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=8,
        )

        for rep in range(n_rep):
            self.update_message(s="rep n. {}".format(rep), message_level=5)
            omega = start_omega
            step = 0

            # create first rep array
            fidelities.append([])
            sys_energies.append([])
            omegas.append([])
            env_energies.append([])

            # append step 0 values
            fidelities[rep].append(sys_fidelity)
            sys_energies[rep].append(sys_energy)
            omegas[rep].append(start_omega)
            env_energies[rep].append(env_energy)

            # set coupler from list
            if coupler_list is not None:
                coupler_index = 0
                self.sys_env_coupling = coupler_list[coupler_index]

            while omega > stop_omega:
                # set alpha and t
                n_qubits = len(self.sys_hamiltonian.qubits)
                alpha = omega / (weaken_coupling * n_qubits)

                # there's not factor of two here, it's all correct
                evolution_time = np.pi / alpha

                # evolve system and reset ancilla
                (
                    sys_fidelity,
                    sys_energy,
                    env_energy,
                    total_density_matrix,
                ) = self.cooling_step(
                    total_density_matrix=total_density_matrix,
                    env_coupling=omega,
                    alpha=alpha,
                    evolution_time=evolution_time,
                )

                # append values
                fidelities[rep].append(sys_fidelity)
                sys_energies[rep].append(sys_energy)
                omegas[rep].append(omega)
                env_energies[rep].append(env_energy)

                epsilon = control_function(
                    omega=omega, t_fridge=env_energy, **ansatz_options
                )
                # if epsilon is zero or some NaN, default to 1000 step linear evolution
                if epsilon == 0:
                    epsilon = 1e-3 * (start_omega - stop_omega)
                omega = omega - epsilon
                step += 1

                if coupler_list is not None:
                    coupler_index += 1
                    self.sys_env_coupling = coupler_list[
                        coupler_index % len(coupler_list)
                    ]

                # print stats on evolution
                self.update_message(
                    has_increased(sys_fidelity, fidelities[rep][-2], "fidelity"),
                    message_level=9,
                )
                self.update_message(
                    has_increased(sys_energy, sys_energies[rep][-2], "energy"),
                    message_level=9,
                )
                self.update_message(
                    "fidelity to gs: {:.4f}, energy diff of traced out rho: {:.4f}".format(
                        sys_fidelity, sys_energy - self.sys_ground_energy
                    ),
                    message_level=5,
                )
                self.update_message(
                    "epsi: {:.3e} prev: {:.3f} fridge E: {:.3e}".format(
                        epsilon, omega + epsilon, env_energy
                    ),
                    message_level=5,
                )
        return fidelities, sys_energies, omegas, env_energies

    def cooling_step(
        self,
        total_density_matrix: np.ndarray,
        alpha: float,
        env_coupling: float,
        evolution_time: float,
    ):
        cooling_hamiltonian = self.cooling_hamiltonian(env_coupling, alpha)

        self.update_message(
            "env coupling value: {}".format(env_coupling), message_level=10
        )
        self.update_message("alpha value: {}".format(alpha), message_level=10)
        self.update_message(
            "evolution_time value: {}".format(evolution_time), message_level=10
        )

        self.update_message("evolving...", message_level=10)
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

        # renormalizing to avoid errors
        traced_density_matrix /= np.trace(traced_density_matrix)
        total_density_matrix /= np.trace(total_density_matrix)

        # computing useful values
        self.update_message("computing values...", message_level=10)
        sys_fidelity = self.sys_fidelity(traced_density_matrix)
        sys_energy = self.sys_energy(traced_density_matrix)
        env_energy = self.env_energy(total_density_matrix)

        # putting the env back in the ground state
        self.update_message("retensoring...", message_level=10)
        total_density_matrix = np.kron(
            traced_density_matrix, self.env_ground_density_matrix
        )
        self.print_msg()
        return sys_fidelity, sys_energy, env_energy, total_density_matrix

    def plot_controlled_cooling(
        self,
        fidelities: list,
        sys_energies: list,
        env_energies: list,
        omegas: list,
        eigenspectrums: list[list],
        suptitle: str = None,
    ):
        plt.rcParams.update(
            {
                "font.family": r"serif",  # use serif/main font for text elements
                "text.usetex": True,  # use inline math for ticks
                "pgf.rcfonts": False,  # don't setup fonts from rc parameters
                "pgf.preamble": "\n".join(
                    [
                        r"\usepackage{url}",  # load additional packages
                        r"\usepackage{lmodern}",  # unicode math setup
                    ]
                ),
                "figure.figsize": (5, 3),
            }
        )
        nrows = 2
        cmap = plt.get_cmap("turbo", len(env_energies))

        fig, axes = plt.subplots(nrows=nrows, figsize=(5, 3))

        plot_temp = False
        ax_bottom = axes[1]

        if plot_temp:
            twin_ax_bottom = ax_bottom.twinx()
        for rep in range(len(env_energies)):
            if rep == 0:
                len_prev = 0
            else:
                len_prev += len(fidelities[rep - 1])
            axes[0].plot(
                range(len_prev, len_prev + len(fidelities[rep])),
                fidelities[rep],
                color=cmap(rep),
                linewidth=2,
            )

            diffs = np.diff(omegas[rep])
            omega_halfs = np.array(omegas[rep][:-1]) + diffs / 2
            y_values = diffs ** (-2)
            ax_bottom.plot(
                omega_halfs,
                y_values,
                color=cmap(rep),
                linewidth=1,
                label="Rep. {}".format(rep + 1),
            )
            if plot_temp:
                twin_ax_bottom.plot(
                    omegas[rep], env_energies[rep], color="red", linewidth=2
                )

        spectrum_cmap = plt.get_cmap("hsv", len(eigenspectrums))
        for ind, spectrum in enumerate(eigenspectrums):
            transitions = get_transition_rates(spectrum)
            ax_bottom.vlines(
                transitions,
                ymin=0,
                ymax=np.nanmax(y_values[np.isfinite(y_values)]),
                linestyle="--",
                color=spectrum_cmap(ind),
                linewidth=0.5,
            )

        axes[0].set_ylabel(r"$|\langle \psi_{cool} | \psi_{gs} \rangle|^2$", labelpad=0)
        axes[0].set_xlabel("step")

        ax_bottom.set_ylabel(r"$(\frac{\mathrm{d}}{\mathrm{ds}}\omega)^{-2}$")
        ax_bottom.tick_params(axis="y")  # , labelcolor="blue")
        ax_bottom.set_yscale("log")
        ax_bottom.invert_xaxis()
        ax_bottom.set_xlabel("Fridge gap")
        if plot_temp:
            twin_ax_bottom.set_ylabel(r"Env. energy")
            twin_ax_bottom.tick_params(axis="y", labelcolor="red")
            twin_ax_bottom.set_yscale("log")

        if suptitle:
            fig.suptitle(suptitle)

        use_cbar = True
        if len(env_energies) > 1:
            if use_cbar:
                fig.colorbar(
                    cm.ScalarMappable(norm=colors.NoNorm(), cmap=cmap), ax=axes
                )
            else:
                ax_bottom.legend(bbox_to_anchor=(0.2, 2))

        plt.show()

    def plot_generic_cooling(
        self,
        sys_energies: list,
        fidelities: list,
        supplementary_data: dict = {},
        suptitle: str = None,
    ):
        nrows = 2 + len(supplementary_data)
        fig, axes = plt.subplots(nrows=nrows, figsize=(5, 3))

        axes[0].plot(range(len(fidelities)), fidelities, color="k", linewidth=2)
        axes[0].set_ylabel(r"$|\langle \psi_{cool} | \psi_{gs} \rangle|^2$", labelpad=0)
        axes[0].set_xlabel(r"$Steps$", labelpad=0)
        axes[1].plot(
            range(len(sys_energies)),
            (np.array(sys_energies) - self.sys_ground_energy)
            / np.abs(self.sys_ground_energy),
            color="k",
            linewidth=2,
        )
        axes[1].set_ylabel(r"$\frac{E_{cool}-E_0}{|E_0|}$", labelpad=0)
        for ind, k in enumerate(supplementary_data.keys()):
            axes[ind + 2].plot(
                range(len(supplementary_data[k])),
                supplementary_data[k],
                color="k",
                linewidth=2,
            )
            axes[ind + 2].set_ylabel(k)

        if suptitle:
            fig.suptitle(suptitle)

        plt.show()


def get_total_spectra_at_given_omega(
    cooler: Cooler,
    Nf: list,
    omega: float,
    weaken_coupling: float,
):
    n_qubits = len(cooler.total_qubits)
    alpha = omega / (weaken_coupling * n_qubits)
    ham = cooler.cooling_hamiltonian(env_coupling=omega, alpha=alpha)
    eigvals, eigvecs = np.linalg.eigh(ham)
    subspace_eigvals = []
    subspace_eigvecs = []
    for eigval, eigvec in zip(eigvals, eigvecs):
        if ket_in_subspace(ket=eigvec, Nf=Nf, n_subspace_qubits=n_qubits // 2):
            subspace_eigvals.append(eigval)
            subspace_eigvecs.append(eigvec)
    return subspace_eigvals, subspace_eigvecs
