import io
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data_manager import ExperimentDataManager, ExtendedJSONDecoder
from fauplotstyle import style

style()
fig, ax = plt.subplots()
experiments = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\single_coupler_improvement",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\single_coupler_improvement_00001",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\single_coupler_improvement_00002",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\single_coupler_improvement_00003",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\single_coupler_improvement_00004",
)

labels = (
    r"Hartree-Fock",
    r"Slater",
    r"Slater + sweep",
    r"Hartree-Fock + Gaussian",
    r"Slater + sweep + Gaussian",
)

for ind, experiment in enumerate(experiments):

    edm = ExperimentDataManager.load(experiment)

    fid_improvements = np.zeros((edm.run_number + 1,))

    for run in range(edm.run_number + 1):
        data_dir = edm.data_dir(run)

        fpath = Path(data_dir, os.listdir(data_dir)[-1])

        jobj = json.loads(
            io.open(fpath, "r", encoding="utf8").read(),
            cls=ExtendedJSONDecoder,
        )
        fid_improvements[run] = jobj["fidelities"][-1][-1] - jobj["fidelities"][-1][0]

    ax.plot(range(len(fid_improvements)), fid_improvements, label=labels[ind])

ax.set_xlabel("Coupler index")
ax.set_ylabel("Fidelity improvement")
ax.legend()


dry_run = False
edm_out = ExperimentDataManager("plot_couplers_single_perf", dry_run=dry_run)

edm_out.var_dump(d={k: v for k, v in zip(labels, experiments)})

edm_out.save_figure(fig, fig_shape="double-size")
plt.show()
