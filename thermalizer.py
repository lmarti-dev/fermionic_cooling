from coolerClass import Cooler
import numpy as np
from utils import time_evolve_density_matrix, trace_out_env
from scipy.linalg import expm
from cirq import fidelity


class Thermalizer(Cooler):
    def __init__(self, beta: float, *args, **kwargs):
        self.beta = beta

        super().__init__(*args, **kwargs)

        # set thermal states
        self.thermal_env_density = expm(
            -self.beta * self.env_hamiltonian.matrix(qubits=self.env_qubits)
        )
        self.thermal_env_density /= np.trace(self.thermal_env_density)
        self.thermal_sys_density = expm(
            -self.beta * self.sys_hamiltonian.matrix(qubits=self.sys_qubits)
        )
        self.thermal_sys_density /= np.trace(self.thermal_sys_density)

    def sys_fidelity(self, state: np.ndarray):
        return fidelity(
            state.astype("complex_"),
            self.thermal_sys_density,
            qid_shape=(2,) * (len(self.sys_qubits)),
        )

    def cooling_step(
        self,
        total_density_matrix: np.ndarray,
        alpha: float,
        env_coupling: float,
        evolution_time: float,
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
        sys_fidelity = self.sys_fidelity(traced_density_matrix)
        sys_energy = self.sys_energy(traced_density_matrix)
        env_energy = self.env_energy(total_density_matrix)

        # putting the env back in the ground state

        total_density_matrix = np.kron(traced_density_matrix, self.thermal_env_density)
        self.print_msg()
        return sys_fidelity, sys_energy, env_energy, total_density_matrix
