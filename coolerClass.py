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

from typing import Iterable, Union

import cirq
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from building_blocks import control_function
from fermionic_cooling.utils import (
    NO_CUPY,
    depth_indexing,
    expectation_wrapper,
    get_list_depth,
    is_density_matrix,
    ketbra,
    time_evolve_density_matrix,
    trace_out_env,
    trace_out_sys,
)

from fauvqe.utilities import (
    flatten,
    ket_in_subspace,
    spin_dicke_mixed_state,
)


class Cooler:
    def __init__(
        self,
        sys_hamiltonian: cirq.PauliSum,
        n_electrons: list,
        sys_qubits: Iterable[cirq.Qid],
        sys_initial_state: np.ndarray,
        sys_ground_state: np.ndarray,
        env_hamiltonian: cirq.PauliSum,
        env_qubits: Iterable[cirq.Qid],
        env_ground_state: np.ndarray,
        sys_env_coupler_data: Union[cirq.PauliSum, list],
        verbosity: int = 0,
        time_evolve_method: str = "diag",
    ):
        if NO_CUPY:
            print("Cupy not installed, using CPU")
            time_evolve_method = "expm"
        self.time_evolve_method = time_evolve_method
        self.verbosity = verbosity
        self.sys_hamiltonian = sys_hamiltonian
        self.n_electrons = n_electrons
        self.sys_qubits = sys_qubits
        self.sys_initial_state = sys_initial_state
        self.sys_ground_state = sys_ground_state
        self._sys_eig_states = None
        self.env_hamiltonian = env_hamiltonian
        self.env_qubits = env_qubits
        self.env_ground_state = env_ground_state
        self.sys_ground_energy = np.real(
            sys_hamiltonian.expectation_from_state_vector(
                sys_ground_state,
                qubit_map={k: v for k, v in zip(sys_qubits, range(len(sys_qubits)))},
            )
        )
        self.sys_env_coupler_data = sys_env_coupler_data

        # check if we got a list to iterate over
        if isinstance(sys_env_coupler_data, cirq.PauliSum):
            self.sys_env_coupler_data_dims = 0
        else:
            self.sys_env_coupler_data_dims = get_list_depth(sys_env_coupler_data)
        self.sys_env_coupler = self.get_coupler_from_data()

        # message for verbose printing
        self.msg = {}
        self.total_cooling_time = None

    def update_message(self, k: str, s: str, message_level: int = 1):
        if int(self.verbosity) >= message_level:
            self.msg[k] = s

    def reset_msg(self):
        self.msg = {}

    def print_msg(self):
        if self.verbosity and self.msg != {}:
            print("\n".join(self.msg.values()))
            print("\033[F" * (1 + len(self.msg.keys())))

    def msg_out(self):
        print("\x1B[#B" * (2 + len(self.msg.keys())))

    def cooling_hamiltonian(self, env_coupling: float, alpha: float):
        if isinstance(self.sys_env_coupler, (cirq.PauliSum, cirq.PauliString)):
            coupler = self.sys_env_coupler.matrix(qubits=self.total_qubits)
        else:
            coupler = self.sys_env_coupler
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
            self.env_qubits,
        )

    def get_coupler_from_data(self, indices: tuple = None):
        # if no index given pick the first element
        if indices is None:
            indices = (0,) * self.sys_env_coupler_data_dims
        return depth_indexing(_list=self.sys_env_coupler_data, indices=iter(indices))

    def zip_cool(
        self,
        evolution_times: np.ndarray,
        alphas: np.ndarray,
        sweep_values: Iterable[float],
        n_rep: int = 1,
        fidelity_threshold: float = 0.9999,
    ):
        initial_density_matrix = self.total_initial_state
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        sys_energies = []
        env_energies = []

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)
        env_energy = self.env_energy(self.env_ground_state)

        self.update_message(
            "fidgs",
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=9,
        )

        fidelities.append(sys_fidelity)
        sys_energies.append(sys_energy)
        env_energies.append(env_energy)
        for rep in range(n_rep):
            self.update_message("repn", f"rep: {rep}")
            for step, env_coupling in enumerate(sweep_values):
                # set one coupler for one gap
                self.sys_env_coupler_easy_setter(coupler_index=step, rep=None)
                (
                    sys_fidelity,
                    sys_energy,
                    env_energy,
                    total_density_matrix,
                ) = self.cooling_step(
                    total_density_matrix=total_density_matrix,
                    env_coupling=env_coupling,
                    alpha=alphas[step],
                    evolution_time=evolution_times[step],
                )
                fidelities.append(sys_fidelity)
                sys_energies.append(sys_energy)
                env_energies.append(env_energy)
                self.update_message(
                    "fidgs",
                    "fidelity to gs: {:.4f}, dE rho_sys: {:.4f} p(1) env {:.3f}".format(
                        sys_fidelity, sys_energy - self.sys_ground_energy, env_energy
                    ),
                    message_level=5,
                )
                self.update_message(
                    "step", f"step: {step} coupling: {env_coupling:.4f}"
                )

                # if we've almost reached fidelity 1, then quit
                if sys_fidelity >= fidelity_threshold:
                    print(f"above fidelity threshold, quitting with: {sys_fidelity}")
                    final_sys_density_matrix = trace_out_env(
                        rho=total_density_matrix,
                        n_sys_qubits=len(self.sys_qubits),
                        n_env_qubits=len(self.env_qubits),
                    )
                    return (
                        fidelities,
                        sys_energies,
                        env_energies,
                        final_sys_density_matrix,
                    )

        final_sys_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )

        self.msg_out()
        return fidelities, sys_energies, env_energies, final_sys_density_matrix

    def loop_cool(
        self,
        evolution_times: np.ndarray,
        alphas: np.ndarray,
        sweep_values: Iterable[float],
        pooling_method: str = "max",
        use_trotter: bool = False,
    ):
        initial_density_matrix = self.total_initial_state
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        sys_energies = []
        env_energies = []

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)
        env_energy = self.env_energy(self.env_ground_state)

        self.update_message(
            "fidgs",
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=9,
        )

        fidelities.append(sys_fidelity)
        sys_energies.append(sys_energy)
        env_energies.append(env_energy)

        for step, env_coupling in enumerate(sweep_values):
            temp_fidelities = []
            temp_sys_energies = []
            temp_env_energies = []
            for coupler_idx in range(self.get_coupler_number(rep=None)):
                self.update_message(
                    "coupn",
                    f"coupler number {coupler_idx}",
                    message_level=5,
                )
                self.sys_env_coupler_easy_setter(coupler_index=coupler_idx, rep=None)
                # trotterize cooling to go slow
                if use_trotter:
                    step_fn = self.trotter_cooling_step
                    evolution_time = 1e-2
                else:
                    step_fn = self.cooling_step
                    evolution_time = evolution_times[step]
                sys_fidelity, sys_energy, env_energy, total_density_matrix = step_fn(
                    total_density_matrix=total_density_matrix,
                    env_coupling=env_coupling,
                    alpha=alphas[step],
                    evolution_time=evolution_time,
                )

                temp_fidelities.append(sys_fidelity)
                temp_sys_energies.append(sys_energy)
                temp_env_energies.append(env_energy)

                self.update_message(
                    "fidgs",
                    "fid. to tgt.: {:.4f}, ΔE of ϱ_sys: {:.4f}".format(
                        sys_fidelity, sys_energy - self.sys_ground_energy
                    ),
                    message_level=5,
                )
                self.update_message(
                    "step", f"step: {step} coupling: {env_coupling:.4f}"
                )

            # since we loop over all couplers, decide how we append the results
            # one per step or all etc.
            if pooling_method == "all":
                fidelities.extend(temp_fidelities)
                sys_energies.extend(temp_sys_energies)
                env_energies.extend(temp_env_energies)
            else:
                if pooling_method == "mean":
                    pool_fn = np.mean
                elif pooling_method == "max":
                    pool_fn = np.max
                elif pooling_method == "min":
                    pool_fn = np.min
                fidelities.append(pool_fn(temp_fidelities))
                sys_energies.append(pool_fn(temp_sys_energies))
                env_energies.append(pool_fn(temp_env_energies))

        final_sys_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )

        self.msg_out()
        return fidelities, sys_energies, env_energies, final_sys_density_matrix

    def sys_env_coupler_easy_setter(self, coupler_index: int, rep: int):
        coupler_indexing = self.sys_env_coupler_data_dims > 0
        dims = self.sys_env_coupler_data_dims
        if coupler_indexing:
            # if there's only one list of couplers, iterate over all of them
            if dims == 1:
                coupler_index_mod = coupler_index % len(self.sys_env_coupler_data)
                coupler_tuple = (coupler_index_mod,)

            # if there are two lists, then iterate over one of them for a rep, then the other for the other rep
            elif dims == 2:
                rep_mod = rep % len(self.sys_env_coupler_data)
                coupler_index_mod = coupler_index % len(
                    self.sys_env_coupler_data[rep_mod]
                )

                # get tupler index
                coupler_tuple = (rep_mod, coupler_index_mod)
            self.sys_env_coupler = self.get_coupler_from_data(coupler_tuple)
        else:
            pass
        self.update_message(
            "coupler",
            f"Current coupler: {self.sys_env_coupler}",
            message_level=7,
        )

    def get_coupler_number(self, rep):
        if self.sys_env_coupler_data_dims == 0:
            coupler_number = 1
        elif self.sys_env_coupler_data_dims == 1:
            coupler_number = len(self.sys_env_coupler_data)
        elif self.sys_env_coupler_data_dims == 2:
            coupler_number = len(self.sys_env_coupler_data[rep])
        return coupler_number

    def big_brain_cool(
        self,
        start_omega: float,
        stop_omega: float,
        n_rep: int = 1,
        ansatz_options: dict = {},
        weaken_coupling: float = 100,
        coupler_transitions: list = None,
        depol_noise: float = None,
        is_noise_spin_conserving: bool = False,
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
        env_energy = self.env_energy(self.env_ground_state)
        n_qubits = len(self.sys_hamiltonian.qubits)

        self.update_message(
            "figs",
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=8,
        )

        self.total_cooling_time = 0
        for rep in range(n_rep):
            self.update_message("rep", s="rep n. {}".format(rep), message_level=5)
            omega = start_omega

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

            # check number of couplers in list
            coupler_number = self.get_coupler_number(rep)
            overall_steps = 0
            while omega > stop_omega:
                self.update_message(
                    "ovstep",
                    f"overall steps: {overall_steps}, total cool. time: {self.total_cooling_time:.2f}",
                )

                # set alpha and t
                alpha = omega / (weaken_coupling * n_qubits)

                # there's not factor of two here, it's all correct
                evolution_time = np.pi / alpha
                self.total_cooling_time += evolution_time

                if coupler_transitions is not None:
                    assert len(coupler_transitions) == coupler_number
                    # cool with couplers for a given gap ONLY IF THEIR ENERGY IS CLOSE
                    coupler_omega_dist = np.abs(coupler_transitions - omega)
                    threshold = 2
                    # [0] because where returns a tuple
                    coupler_indices = np.where(coupler_omega_dist <= threshold)[0]
                    if len(coupler_indices) == 0:
                        # either we use all coupler in omega steppes or none
                        coupler_indices = range(coupler_number)
                        coupler_indices = []
                else:
                    # cool with all couplers for a given gap
                    coupler_indices = range(coupler_number)
                measured_env_energies = []
                for coupler_index in coupler_indices:
                    overall_steps += 1
                    self.update_message(
                        "cind",
                        f"Eff. num. of coupler: {len(coupler_indices):.1f}, coupler {coupler_index:.1f}",
                    )
                    self.sys_env_coupler_easy_setter(
                        coupler_index=coupler_index, rep=rep
                    )

                    reset = False
                    if coupler_index == coupler_indices[-1]:
                        reset = True
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
                        do_reset_fridge=reset,
                        depol_noise=depol_noise,
                        is_noise_spin_conserving=is_noise_spin_conserving,
                    )

                    # print stats on evolution
                    self.update_message(
                        "fidgs",
                        "fid. to tgt.: {:.4f}, ΔE of ϱ_sys: {:.4f}".format(
                            sys_fidelity, sys_energy - self.sys_ground_energy
                        ),
                        message_level=5,
                    )
                    measured_env_energies.append(env_energy)

                # make sure that we update the control func with the highest env_energy
                # becuase we are concerned with the best coupler
                # if one coupler works we better slow down
                if measured_env_energies:
                    env_energy = np.max(measured_env_energies)
                else:
                    env_energy = 0
                # append values
                fidelities[rep].append(sys_fidelity)
                sys_energies[rep].append(sys_energy)
                omegas[rep].append(omega)
                env_energies[rep].append(env_energy)

                # update the control function
                mean_env_energy = np.mean(env_energies[rep])
                epsilon = control_function(
                    omega=omega,
                    t_fridge=env_energy,
                    t_mean=mean_env_energy,
                    **ansatz_options,
                )
                # if epsilon is zero or some NaN, default to 1000 step linear evolution
                if epsilon == 0:
                    epsilon = 1e-6 * (start_omega - stop_omega)
                self.update_message(
                    "epsi",
                    f"ε: {epsilon:.3e} ω: {omega:.3f} Z_fr: {env_energy/omega:.3e} start: {start_omega:.3f} stop: {stop_omega:.3f}",
                    message_level=5,
                )

                omega = omega - epsilon

        final_sys_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )
        self.msg_out()
        return fidelities, sys_energies, omegas, env_energies, final_sys_density_matrix

    def trotter_cooling_step(
        self,
        total_density_matrix: np.ndarray,
        alpha: float,
        env_coupling: float,
        evolution_time: float = 1e-3,
    ):
        cooling_hamiltonian = self.cooling_hamiltonian(env_coupling, alpha)

        # set up dummy values
        total_time = 0
        sys_energy = 1e10
        sys_energy_before = sys_energy
        sys_energy_after = 0
        while sys_energy_after < sys_energy_before:
            sys_energy_before = sys_energy
            total_density_matrix = time_evolve_density_matrix(
                ham=cooling_hamiltonian,  # .matrix(qubits=self.total_qubits),
                rho=total_density_matrix,
                t=evolution_time,
                method="expm",
            )
            total_time += evolution_time
            traced_density_matrix = trace_out_env(
                rho=total_density_matrix,
                n_sys_qubits=len(self.sys_qubits),
                n_env_qubits=len(self.env_qubits),
            )
            traced_env = trace_out_sys(
                rho=total_density_matrix,
                n_sys_qubits=len(self.sys_qubits),
                n_env_qubits=len(self.env_qubits),
            )

            # renormalizing to avoid errors
            traced_density_matrix /= np.trace(traced_density_matrix)
            total_density_matrix /= np.trace(total_density_matrix)
            traced_env /= np.trace(traced_env)

            sys_energy = self.sys_energy(traced_density_matrix)
            sys_energy_after = sys_energy

            self.update_message(
                "trottime",
                f"total evolved time [α/π]: {total_time*alpha/np.pi:.5f}, energy: {sys_energy:.4f}",
                message_level=5,
            )
            self.print_msg()
        sys_fidelity = self.sys_fidelity(traced_density_matrix)
        env_energy = self.env_energy(traced_env)
        # putting the env back in the ground state
        total_density_matrix = np.kron(
            traced_density_matrix, self.env_ground_density_matrix
        )

        return sys_fidelity, sys_energy, env_energy, total_density_matrix

    def cooling_step(
        self,
        total_density_matrix: np.ndarray,
        alpha: float,
        env_coupling: float,
        evolution_time: float,
        depol_noise: float = None,
        do_reset_fridge: bool = True,
        is_noise_spin_conserving: bool = False,
    ):
        cooling_hamiltonian = self.cooling_hamiltonian(env_coupling, alpha)

        self.update_message(
            "envcou", "env coupling value: {}".format(env_coupling), message_level=10
        )
        self.update_message("alpha", "alpha value: {}".format(alpha), message_level=10)
        self.update_message(
            "evtime",
            "evolution_time value: {}".format(evolution_time),
            message_level=10,
        )

        total_density_matrix = time_evolve_density_matrix(
            ham=cooling_hamiltonian,  # .matrix(qubits=self.total_qubits),
            rho=total_density_matrix,
            t=evolution_time,
            method=self.time_evolve_method,
        )

        traced_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )
        traced_env = trace_out_sys(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )

        if depol_noise is not None:
            # add depol noise

            if is_noise_spin_conserving is True:
                rho_err = spin_dicke_mixed_state(
                    n_qubits=len(self.sys_qubits), Nf=self.n_electrons, expanded=True
                )
            else:
                rho_err = np.eye(len(traced_density_matrix))

            traced_density_matrix = (
                1 - depol_noise
            ) * traced_density_matrix + depol_noise * rho_err

        # renormalizing to avoid errors
        traced_density_matrix /= np.trace(traced_density_matrix)
        total_density_matrix /= np.trace(total_density_matrix)
        traced_env /= np.trace(traced_env)

        # computing useful values
        sys_fidelity = self.sys_fidelity(traced_density_matrix)
        sys_energy = self.sys_energy(traced_density_matrix)
        env_energy = self.env_energy(traced_env)

        self.print_msg()

        if not do_reset_fridge:
            return sys_fidelity, sys_energy, env_energy, total_density_matrix

        # putting the env back in the ground state
        total_density_matrix = np.kron(
            traced_density_matrix, self.env_ground_density_matrix
        )
        return sys_fidelity, sys_energy, env_energy, total_density_matrix

    @classmethod
    def plot_controlled_cooling(
        cls,
        fidelities: list,
        env_energies: list,
        omegas: list,
        eigenspectrums: list[list],
        suptitle: str = None,
    ):
        more_than_one_rep = len(env_energies) > 1
        nrows = 2
        if more_than_one_rep:
            try:
                cmap_name = "faucmap"
                cmap = plt.get_cmap(cmap_name, len(env_energies))
            except Exception:
                cmap_name = "turbo"
                cmap = plt.get_cmap(cmap_name, len(env_energies))
                print(f"using {cmap_name}")

        fig, axes = plt.subplots(nrows=nrows, sharex=True)

        plot_temp = False
        ax_bottom = axes[1]
        spectrum_cmap = plt.get_cmap("hsv", len(eigenspectrums))
        if plot_temp:
            twin_ax_bottom = ax_bottom.twinx()
        for rep in range(len(env_energies)):
            if rep == 0:
                len_prev = 0
                sum_omega = 0
            else:
                len_prev += len(fidelities[rep - 1])
                sum_omega += sum(omegas[rep - 1])
            if more_than_one_rep:
                kwargs = {"color": cmap(rep)}
            else:
                kwargs = {}
            axes[0].plot(omegas[rep], fidelities[rep], **kwargs)

            ax_bottom.plot(
                omegas[rep],
                env_energies[rep],
                **kwargs,
            )
        all_env_energies = np.array(list(flatten(env_energies)))
        for ind, spectrum in enumerate(eigenspectrums):
            ax_bottom.vlines(
                spectrum,
                ymin=0,
                ymax=np.nanmax(all_env_energies[np.isfinite(all_env_energies)]),
                linestyle="--",
                color=spectrum_cmap(ind),
            )
        axes[0].set_ylabel(r"Fidelity", labelpad=0)

        ax_bottom.set_ylabel(r"$T_F$")
        ax_bottom.tick_params(axis="y")
        ax_bottom.set_yscale("log")
        ax_bottom.invert_xaxis()
        ax_bottom.set_xlabel("$\omega$")
        if plot_temp:
            twin_ax_bottom.set_ylabel(r"Env. energy")
            twin_ax_bottom.tick_params(axis="y", labelcolor="red")
            twin_ax_bottom.set_yscale("log")

        if suptitle:
            fig.suptitle(suptitle)

        use_cbar = True
        if more_than_one_rep:
            if use_cbar:
                fig.colorbar(
                    cm.ScalarMappable(norm=colors.NoNorm(), cmap=cmap), ax=axes
                )
            else:
                ax_bottom.legend(bbox_to_anchor=(0.2, 2))
        return fig

    @classmethod
    def plot_default_cooling(
        cls,
        omegas: np.ndarray,
        fidelities: np.ndarray,
        env_energies: np.ndarray,
        suptitle: str,
        n_rep: int = None,
    ):
        fig, axes = plt.subplots(nrows=2, sharex=True)

        if n_rep is not None and len(omegas) % n_rep == 0 and n_rep > 1:
            slice_size = int(len(omegas) / n_rep)
            for rep in range(n_rep):
                idx = slice(rep * slice_size, (rep + 1) * slice_size)
                axes[0].plot(
                    omegas[idx],
                    fidelities[idx],
                )
                axes[1].plot(
                    omegas[idx],
                    env_energies[idx],
                )

                if rep < n_rep - 1:
                    axes[0].plot(
                        [
                            omegas[(rep + 1) * slice_size - 1],
                            omegas[(rep + 1) * slice_size],
                        ],
                        [
                            fidelities[(rep + 1) * slice_size - 1],
                            fidelities[(rep + 1) * slice_size],
                        ],
                        "k:",
                    )
        else:
            axes[0].plot(
                omegas,
                fidelities,
            )
            axes[1].plot(
                omegas,
                env_energies,
            )
        axes[0].set_ylabel("Fidelity")
        axes[1].set_ylabel(r"$T_F$")
        axes[1].set_xlabel(r"$\omega$")
        axes[1].invert_xaxis()

        if suptitle:
            fig.suptitle(suptitle)

        return fig

    @classmethod
    def plot_generic_cooling(
        cls,
        fidelities: list,
        initial_pops: list,
        env_energies: list,
        n_rep: int,
        supplementary_data: dict = {},
        suptitle: str = None,
    ):
        nrows = 2 + len(supplementary_data)
        fig, axes = plt.subplots(nrows=nrows)

        axes[0].plot(
            range(len(fidelities)),
            fidelities,
            "kx--",
        )
        axes[0].set_ylabel(r"$|\langle \psi_{cool} | \psi_{gs} \rangle|^2$")
        axes[0].set_xlabel(r"$Steps$")
        cmap = plt.get_cmap("turbo", n_rep)
        for ind in range(n_rep - 1):
            axes[1].plot(
                range(
                    len(
                        env_energies[
                            ind * len(initial_pops) : (ind + 1) * len(initial_pops)
                        ]
                    )
                ),
                env_energies[ind * len(initial_pops) : (ind + 1) * len(initial_pops)],
                color=cmap(ind),
            )
        axes[1].plot(
            range(len(initial_pops)),
            list(reversed(initial_pops)),
            color="k",
            linestyle=":",
            label="initial pop.",
        )
        axes[1].legend()
        if supplementary_data:
            for ind, k in enumerate(supplementary_data.keys()):
                axes[ind + 2].plot(
                    range(len(supplementary_data[k])),
                    supplementary_data[k],
                    color="k",
                )
                axes[ind + 1].set_ylabel(k)

        if suptitle:
            fig.suptitle(suptitle)

        return fig

    def probe_evolution_times(self, alphas, sweep_values, N_slices: int):
        initial_density_matrix = self.total_initial_state
        if not cirq.is_hermitian(initial_density_matrix):
            raise ValueError("initial density matrix is not hermitian")
        total_density_matrix = initial_density_matrix

        fidelities = []
        energies = []

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)

        self.update_message(
            "fidgs",
            "initial fidelity to gs: {:.4f}, initial energy of traced out rho: {:.4f}, ground energy: {:.4f}".format(
                sys_fidelity, sys_energy, self.sys_ground_energy
            ),
            message_level=9,
        )

        fidelities.append(sys_fidelity)
        energies.append(sys_energy)
        env_energy_dynamics = []
        for step, env_coupling in enumerate(sweep_values):
            max_evolution_time = 4 * np.pi / alphas[step]
            evolution_times = []
            env_energies = []
            for evolution_time in np.linspace(
                max_evolution_time / N_slices, max_evolution_time, N_slices
            ):
                (
                    sys_fidelity,
                    sys_energy,
                    env_energy,
                    _,
                ) = self.cooling_step(
                    total_density_matrix=total_density_matrix,
                    env_coupling=env_coupling,
                    alpha=alphas[step],
                    evolution_time=evolution_time,
                )
                fidelities.append(sys_fidelity)
                energies.append(sys_energy)
                self.update_message(
                    "evtime",
                    f"evolution time: {evolution_time:.3f}/{max_evolution_time:.3f}, env_ene: {env_energy:.3f}",
                    message_level=5,
                )
                self.update_message(
                    "fidgs",
                    "fid. to tgt.: {:.4f}, ΔE of ϱ_sys: {:.4f}".format(
                        sys_fidelity, sys_energy - self.sys_ground_energy
                    ),
                    message_level=5,
                )
                self.update_message(
                    "step", f"step: {step} coupling: {env_coupling:.4f}"
                )
                evolution_times.append(evolution_time)
                env_energies.append(env_energy)
            env_energy_dynamics.append((evolution_times, env_energies))
        final_sys_density_matrix = trace_out_env(
            rho=total_density_matrix,
            n_sys_qubits=len(self.sys_qubits),
            n_env_qubits=len(self.env_qubits),
        )
        return fidelities, energies, final_sys_density_matrix, env_energy_dynamics


def get_total_spectra_at_given_omega(
    cooler: Cooler,
    n_electrons: list,
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
        if ket_in_subspace(
            ket=eigvec, n_electrons=n_electrons, n_subspace_qubits=n_qubits // 2
        ):
            subspace_eigvals.append(eigval)
            subspace_eigvecs.append(eigvec)
    return subspace_eigvals, subspace_eigvecs
