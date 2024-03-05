from coolerClass import Cooler
import numpy as np
from fermionic_cooling.utils import time_evolve_density_matrix, trace_out_env
from fauvqe.utilities import spin_dicke_mixed_state
from cirq import fidelity


class Thermalizer(Cooler):
    def __init__(
        self,
        beta: float,
        thermal_env_density: np.ndarray,
        thermal_sys_density: np.ndarray,
        *args,
        **kwargs
    ):

        self.beta = beta

        super().__init__(*args, **kwargs)

        # set thermal states
        self.thermal_env_density = thermal_env_density
        self.thermal_sys_density = thermal_sys_density

        self.sys_ground_energy = np.real(
            self.sys_hamiltonian.expectation_from_density_matrix(
                self.thermal_sys_density,
                qubit_map={
                    k: v for k, v in zip(self.sys_qubits, range(len(self.sys_qubits)))
                },
            )
        )

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

        self.print_msg()

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

        if not do_reset_fridge:
            return sys_fidelity, sys_energy, env_energy, total_density_matrix

        # putting the env back in the thermal state
        total_density_matrix = np.kron(traced_density_matrix, self.thermal_env_density)

        return sys_fidelity, sys_energy, env_energy, total_density_matrix
