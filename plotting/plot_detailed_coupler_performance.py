import matplotlib.pyplot as plt
from fauplotstyle.styler import style
from data_manager import ExperimentDataManager, ExtendedJSONDecoder
import json
import io
import os
import matplotlib.cm as cm
import matplotlib.colors as colors


def get_var_dump(logging_dir: str):
    files = os.listdir(logging_dir)
    var_dump = [x for x in files if x.startswith("var_dump")]
    jobj = json.loads(
        io.open(os.path.join(logging_dir, var_dump[0]), "r", encoding="utf8").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


def get_data(data_dir: str):
    files = os.listdir(data_dir)
    fpath = os.path.join(data_dir, files[0])
    print(f"loading {fpath}")
    jobj = json.loads(
        io.open(fpath, "r", encoding="utf8").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


style()
dry_run = False
edm = ExperimentDataManager(
    experiment_name="plot_couplers_oneatatime",
    notes="plotting the couplers individual resonances",
    dry_run=dry_run,
)

dirname = "/home/eckstein/Desktop/projects/data/2024_02_23/fh22_oneatatime_09h25/"

runs = list(sorted(os.listdir(dirname)))
cmap = plt.get_cmap("faucmap", len(runs))

fig, axes = plt.subplots(nrows=2, sharex=True)

for ind, run in enumerate(runs):
    data_dir = os.path.join(dirname, run, "data/")
    data = get_data(data_dir)
    omegas = data["omegas"]
    fidelities = data["fidelities"]
    env_energies = data["env_energies"]
    axes[0].plot(omegas[0], fidelities[0], color=cmap(ind), label=f"Coupler {ind}")
    axes[1].plot(
        omegas[0],
        env_energies[0],
        color=cmap(ind),
    )
axes[1].invert_xaxis()
cbar = fig.colorbar(cm.ScalarMappable(norm=colors.NoNorm(), cmap=cmap), ax=axes)
cbar.set_label("Coupler index")

axes[0].set_ylabel("Fidelity")
axes[1].set_ylabel("$E_F$")
axes[1].set_xlabel("$\omega$")

edm.save_figure(fig=fig, filename="plot_each_coupler_resonance", add_timestamp=False)
plt.show()
