import numpy as np
from qutlet.utilities import jw_get_free_couplers, pretty_str_pauli_sum
from qutlet.models import FermiHubbardModel


from data_manager import ExperimentDataManager


def gaussian_bell(mu: float, sigma: float, n_steps: int):
    return np.exp(-((np.linspace(-1, 1, n_steps) - mu) ** 2) / (2 * sigma**2))


model = FermiHubbardModel(
    lattice_dimensions=(2, 2), tunneling=1, coulomb=2, n_electrons="half-filling"
)

couplers = jw_get_free_couplers(model=model)

weights = gaussian_bell(1 / 2, 1, len(couplers))
coupler = sum([x * w for x, w in zip(couplers, weights)])


edm = ExperimentDataManager(
    "free_coupler_gaussian_sum",
    notes="summing the paulisum with a gaussian and looking at the resulting_term",
)

edm.var_dump(
    model=model.__to_json__,
    coupler=coupler,
    pretty_coupler=pretty_str_pauli_sum(coupler, n_qubits=model.n_qubits),
)
