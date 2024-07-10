from cooler_class import Cooler
import numpy as np

import matplotlib.pyplot as plt

n_steps = 1000

fidelities = [np.random.rand(n_steps)]
env_ev_energies = [np.random.rand(n_steps) - 0.5]
omegas = [np.linspace(10, 0.4, n_steps)]

eigenspectrum = [-10, 0, 1, 1, 1, 1, 11, 4, 5, 8, 20, 40, 100]

fig = Cooler.plot_controlled_cooling(
    fidelities=fidelities,
    env_energies=env_ev_energies,
    omegas=omegas,
    eigenspectrums=[
        eigenspectrum,
    ],
)

plt.show()
