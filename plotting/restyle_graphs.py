from data_manager import load_figure_data
import matplotlib.pyplot as plt

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from openfermion import get_sparse_operator
import numpy as np
from fau_colors import colors_dark
from data_manager import ExperimentDataManager


def get_values(ax: plt.Axes):
    y = ax.lines[0]._y
    x = ax.lines[0]._x
    return (x, y)


dry_run = False
edm = ExperimentDataManager(experiment_name="figure_restyling", dry_run=dry_run)

model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
n_electrons = [2, 2]
eigenenergies, eigenstates = jw_eigenspectrum_at_particle_number(
    get_sparse_operator(model.fock_hamiltonian), particle_number=n_electrons
)


fpaths = [
    r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graphs\cooling_graphs\depol_1e8\citrus_data_2023_12_13_15_31_07.json",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graphs\cooling_graphs\depol_1e5\limes_data_2023_12_15_07_01_02.json",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graphs\cooling_graphs\depol_fermion_1e5\date_data_2023_12_18_17_32_08.json",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graphs\cooling_graphs\nice_cooling\cherry_data_2023_12_12_14_17_19.json",
]


for fpath in fpaths:
    fig = load_figure_data(fpath)

    axes = fig.get_axes()

    xvt, yvt = get_values(axes[0])
    xvb, yvb = get_values(axes[1])

    yvb = yvb / (np.nanmax(yvb[np.isfinite(yvb)]) - np.nanmin(yvb[np.isfinite(yvb)]))

    new_fig, new_axes = plt.subplots(nrows=2, sharex=True)

    new_axes[0].plot(
        xvb,
        yvt[:-1],
        color=colors_dark.nat,
        linewidth=1.5,
    )

    new_axes[1].clear()
    new_axes[1].vlines(
        eigenenergies - eigenenergies[0],
        ymin=0,
        ymax=np.nanmax(yvb[np.isfinite(yvb)]),
        linestyle="--",
        color="r",
        linewidth=1,
    )
    new_axes[1].plot(
        xvb,
        yvb,
        color=colors_dark.nat,
        linewidth=1.5,
    )
    new_axes[1].set_yscale("log")
    new_axes[1].invert_xaxis()

    new_axes[0].set_ylabel("Fidelity")
    new_axes[1].set_xlabel("$\omega$")
    new_axes[1].set_ylabel(r"$f(E_{fridge})^{-2}$")

    figname = fpath.split("\\")[-2]
    edm.save_figure(fig=new_fig, filename=figname)
