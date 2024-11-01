import cirq
import numpy as np
from adiabatic_sweep import get_sweep_hamiltonian, run_sweep
from scipy.linalg import expm
from cooler_class import Cooler
from fermionic_cooling.utils import trace_out_env, time_evolve_density_matrix


def get_trotterized_sweep_cooling_unitaries(
    sweep_hamiltonian: callable, n_steps: int, total_time: float, alpha
):
    for step in range(1, n_steps + 2):
        unitary = expm(
            -1j * (total_time / n_steps) * sweep_hamiltonian(step / (n_steps + 1))
        )
        yield unitary


class AdiabaticCooler(Cooler):
    def __init__(
        self,
        ham_start: cirq.PauliSum,
        ham_stop: cirq.PauliSum,
        *args,
        **kwargs,
    ):
        self.ham_start = ham_start
        self.ham_stop = ham_stop
        self.sweep_hamiltonian: callable = get_sweep_hamiltonian(
            ham_start=ham_start, ham_stop=ham_stop
        )

        super().__init__(*args, **kwargs)

    def set_sys_sweep_hamiltonian(self, step: int):
        self.sys_hamiltonian = self.sweep_hamiltonian(step)

    def cooling_hamiltonian(self, step: int, env_coupling: float, alpha: float):
        if isinstance(self.sys_env_coupler, (cirq.PauliSum, cirq.PauliString)):
            coupler = self.sys_env_coupler.matrix(qubits=self.total_qubits)
        else:
            coupler = self.sys_env_coupler
        self.set_sys_sweep_hamiltonian(step)
        return (
            self.sys_hamiltonian.matrix(qubits=self.total_qubits)
            + env_coupling * self.env_hamiltonian.matrix(qubits=self.total_qubits)
            + float(alpha) * coupler
        )

    def adiabatic_sweep(
        self,
        total_time: float,
        n_steps: int,
        instantaneous_ground_states: np.ndarray = None,
    ):
        # TODO: either get rid of inst gs, or compute them directly
        # current solution is ugly
        ham_start = self.ham_start.matrix(self.sys_qubits)
        ham_stop = self.ham_stop.matrix(self.sys_qubits)
        print(f"Simulating for {total_time} time and {n_steps} steps")
        fidelities, instant_fidelities, final_ground_state = run_sweep(
            initial_state=self.sys_initial_state,
            ham_start=ham_start,
            ham_stop=ham_stop,
            final_ground_state=self.sys_ground_state,
            instantaneous_ground_states=instantaneous_ground_states,
            n_steps=n_steps,
            total_time=total_time,
        )
        return fidelities, instant_fidelities, final_ground_state

    def adiabatic_cool(
        self,
        evolution_times: list,
        alphas: list,
        omegas: list,
    ):
        total_density_matrix = self.total_initial_state

        sys_fidelities = []
        sys_energies = []
        env_energies = []

        sys_fidelity = self.sys_fidelity(self.sys_initial_state)
        sys_energy = self.sys_energy(self.sys_initial_state)
        env_energy = self.env_energy(total_density_matrix)

        sys_fidelities.append(sys_fidelity)
        sys_energies.append(sys_energy)
        env_energies.append(env_energy)

        total_time = sum(evolution_times)
        self.sys_env_coupler_easy_setter(coupler_index=0, rep=None)
        print("cooling adiabatically")
        for step, time, omega, alpha in zip(
            range(len(evolution_times)),
            evolution_times,
            omegas,
            alphas,
        ):
            total_density_matrix = time_evolve_density_matrix(
                ham=self.cooling_hamiltonian(
                    step / (len(evolution_times) - 1), env_coupling=omega, alpha=alpha
                ),
                rho=total_density_matrix,
                t=time,
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
            sys_fidelity = self.sys_fidelity(traced_density_matrix)
            sys_energy = self.sys_energy(traced_density_matrix)
            env_energy = self.env_energy(total_density_matrix)

            print(
                f"step: {step} time {sum(evolution_times[:step]):.3f}/{total_time:.3f} fid. {sys_fidelity:.3f} env. ene:{env_energy/omega:.3f}",
                end="\r",
            )

            # putting the env back in the ground state
            total_density_matrix = np.kron(
                traced_density_matrix, self.env_ground_density_matrix
            )
            sys_fidelities.append(sys_fidelity)
            sys_energies.append(sys_energy)
            env_energies.append(env_energy)

        return sys_fidelities, sys_energies, env_energies, total_density_matrix
