from cirq import Circuit, SimulatorBase, PauliSum
from qutlet import Ansatz
from qutlet.models import FermionicModel
from fermionic_cooling.cooler_class import Cooler
from fermionic_cooling.utils import ketbra
import numpy as np


# this is what it feels to create Frankenstein
class CoolingAnsatz(Ansatz, Cooler):
    def __init__(
        self,
        model: FermionicModel,
        sys_hamiltonian: np.ndarray,
        subspace_simulation: bool,
        sys_initial_state: np.ndarray,
        env_hamiltonian: PauliSum,
        env_ground_state: np.ndarray,
        env_qubits: int,
        sys_env_coupler_data: list,
        symbols: list = None,
        params: list = None,
        weaken_coupling: int = 60,
        verbosity: int = 5,
        time_evolve_method: str = "expm",
        return_env_energies: bool = False,
    ) -> None:
        if subspace_simulation:
            sys_ground_energy, sys_ground_state = model.subspace_gs
        else:
            sys_ground_energy, sys_ground_state = model.gs

        sys_eig_energies, _ = model.subspace_spectrum
        self.spectrum_width = max(sys_eig_energies) - min(sys_eig_energies)

        self.weaken_coupling = weaken_coupling
        self.model = model

        self.picked_couplers = []
        self.env_cooled_energies = []

        Cooler.__init__(
            self,
            sys_hamiltonian=sys_hamiltonian,
            n_electrons=model.n_electrons,
            sys_qubits=model.qubits,
            sys_ground_state=sys_ground_state,
            sys_initial_state=sys_initial_state,
            env_hamiltonian=env_hamiltonian,
            env_qubits=env_qubits,
            env_ground_state=env_ground_state,
            sys_env_coupler_data=sys_env_coupler_data,
            verbosity=verbosity,
            subspace_simulation=subspace_simulation,
            time_evolve_method=time_evolve_method,
        )
        Ansatz.__init__(
            self, circuit=None, simulator=None, symbols=symbols, params=params
        )

    def simulate(self, opt_params: list = None, initial_state: np.ndarray = None):
        self.env_cooled_energies = []
        if opt_params is None:
            opt_params = self.params
        if initial_state is None:
            initial_state = self.sys_initial_state

        if len(initial_state.shape) == 1:
            initial_state = ketbra(initial_state)
        total_density_matrix = np.kron(initial_state, ketbra(self.env_ground_state))

        for ind in range(len(self.picked_couplers)):
            env_coupling = self.params[ind] * self.spectrum_width
            self.sys_env_coupler = self.picked_couplers[ind]
            alpha = env_coupling / (self.weaken_coupling * self.model.n_qubits)
            evolution_time = np.pi / alpha

            (
                sys_cooled_fidelity,
                sys_cooled_energy,
                env_cooled_energy,
                total_density_matrix,
            ) = self.cooling_step(
                total_density_matrix=total_density_matrix,
                alpha=alpha,
                env_coupling=env_coupling,
                evolution_time=evolution_time,
            )
            self.env_cooled_energies.append(env_cooled_energy)
        return total_density_matrix


def cooled_env_energies_objective(cooler: CoolingAnsatz):
    def objective(env_cooled_energies: np.ndarray) -> float:
        return -np.sum(cooler.env_cooled_energies)

    return objective


def cooling_energy_objective(cooler: CoolingAnsatz, which: str = "env") -> callable:
    if which == "env":
        trace_out = "sys"
        fn = cooler.env_energy
    elif which == "sys":
        trace_out = "env"
        fn = cooler.sys_energy

    def objective(state: np.ndarray) -> float:
        state = cooler.partial_trace_wrapper(state, trace_out=trace_out)
        return -fn(state)

    return objective


def cooling_infidelity_objective(cooler: CoolingAnsatz) -> callable:
    def objective(state: np.ndarray) -> float:
        state = cooler.partial_trace_wrapper(state, trace_out="env")
        return 1 - cooler.sys_fidelity(state)

    return objective
