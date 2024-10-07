from fermionic_cooling.filter_functions import get_fourier_gaps_filter_function
from qutlet.models import FermiHubbardModel


model = FermiHubbardModel(
    lattice_dimensions=(2, 2), tunneling=1, coulomb=6, n_electrons="hf"
)
energies, states = model.spectrum
