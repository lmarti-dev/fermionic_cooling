from data_manager import ExperimentDataManager

import matplotlib.pyplot as plt

import numpy as np

from fauplotstyle import style


fpath = r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_18\time_dependent_coupler_cooling_save_comps"

style()
edm = ExperimentDataManager.load(fpath)

d = edm.load_last_saved_data_file


fig, ax = plt.subplots()
ax: plt.Axes
arr = np.array(d["eig_components"])

cmap = plt.get_cmap("turbo", arr.shape[1])

for ind, comp in enumerate(arr.T):
    ax.plot(range(len(comp)), comp, color=cmap(ind), label=str(ind))
    ax.text(75, comp[75], s=str(ind), color=cmap(ind))

ax.set_ylabel("Amplitude")
ax.set_xlabel("Repetition")
ax.set_yscale("log")


edm_out = ExperimentDataManager(
    "plot_comp_evolution_with_reps",
    notes="Plotting the evolution of eigencomps of the density matrix as the reps go by",
    project="fermionic cooling",
)

edm_out.var_dump(fpath=fpath)
edm_out.save_figure(fig, fig_shape="double-size")

plt.show()
