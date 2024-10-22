import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from data_manager import ExtendedJSONDecoder
from utils import get_most_recent_timestamped_files

from data_manager import ExperimentDataManager
from fauplotstyle.styler import style
import matplotlib.cm as cm
import matplotlib.colors as colors

dry_run = False
edm = ExperimentDataManager(
    experiment_name="plot_gs_index_fidelity",
    notes="just plotting the results of the big double for loop",
    tags="plot,fermions,gs_index,",
    project="fermionic cooling",
    dry_run=dry_run,
)


data_dirname = edm.data_folder


experiment_path = os.path.join(
    edm.data_folder, "2024_03_28/", "fh_bigbrain_subspace_gs_index_09h10/"
)


gs_ind_final_fids = np.zeros((7, 7))

for run in [s for s in os.listdir(experiment_path) if not s.startswith("__")]:
    print(run)
    run_dirname = os.path.join(experiment_path, run)
    log_dirname = os.path.join(run_dirname, "logging/")
    data_dirname = os.path.join(run_dirname, "data/")
    files = [
        os.path.join(log_dirname, x) for x in os.listdir(log_dirname) if "var_dump" in x
    ]
    latest_var_dump = get_most_recent_timestamped_files(files)
    jobj = json.loads(
        io.open(latest_var_dump, "r", encoding="utf8").read(), cls=ExtendedJSONDecoder
    )
    cgs = jobj["coupler_gs_index"]
    sgs = jobj["start_gs_index"]

    data_file = os.listdir(os.path.join(run_dirname, "data/"))[-1]
    jobj = json.loads(
        io.open(os.path.join(data_dirname, data_file), "r", encoding="utf8").read(),
        cls=ExtendedJSONDecoder,
    )

    gs_ind_final_fids[sgs, cgs] = jobj["fidelities"][0][-1]

style()
cmap = plt.get_cmap("faucmap", 7)

scalar_mappable = cm.ScalarMappable(cmap=cmap)
scalar_mappable.set_array(gs_ind_final_fids)
scalar_mappable.autoscale()

fig, ax = plt.subplots()

ax.imshow(gs_ind_final_fids, cmap=cmap)
ax.set_xlabel(r"Coupler index $i:=|\tilde{E}_i\rangle\langle \tilde{E}_n|$")
ax.set_ylabel(r"Start index $i:=|\tilde{E}_i\rangle$")
fig.colorbar(scalar_mappable, ax=ax)

edm.save_dict({"fidelities": gs_ind_final_fids})
edm.save_figure(fig)
plt.show()
