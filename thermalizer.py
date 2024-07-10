from cooler_class import Cooler
import numpy as np
from fermionic_cooling.utils import fidelity_wrapper


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

        self.sys_ground_energy = self.sys_energy(self.thermal_sys_density)

    def sys_fidelity(self, state: np.ndarray):
        return fidelity_wrapper(
            a=state,
            b=self.thermal_sys_density,
            qid_shape=(2,) * len(self.sys_qubits),
            subspace_simulation=self.subspace_simulation,
        )

    @property
    def env_ground_density_matrix(self):
        return self.thermal_env_density
