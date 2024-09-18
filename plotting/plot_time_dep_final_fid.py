from data_manager import ExperimentDataManager, ExtendedJSONDecoder
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import io
import numpy as np

from fauplotstyle import style

style()

fig, ax = plt.subplots()


experiments = (
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_16\time_dependent_coupler_cooling_00004",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_17\time_dependent_coupler_cooling",
    r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_09_17\time_dependent_coupler_cooling_00001",
)
labels = (
    "free couplers zoom",
    "free couplers no filter",
    "free couplers no filter zoom",
)


for exp, lab in zip(experiments, labels):

    edm = ExperimentDataManager.load(exp)

    fidelities = np.zeros((edm.run_number,))
    times = np.zeros((edm.run_number,))

    for run in range(edm.run_number):

        fpath = Path(edm.data_dir(run), os.listdir(edm.data_dir(run))[-1])
        jobj = json.loads(
            io.open(fpath, "r", encoding="utf8").read(),
            cls=ExtendedJSONDecoder,
        )

        fidelities[run] = jobj["total_fidelities"][-1]

        var_dump = edm.load_var_dump(run)

        times[run] = var_dump["total_sim_time"]

    ax.plot(times, fidelities, label=lab)

ax.set_xlabel("Cooling step time")
ax.set_ylabel("Final fidelity")
ax.legend()


edm = ExperimentDataManager("plot_sim_time_fidelities", dry_run=False)

edm.save_dict({"times": times, "fidelities": fidelities})

edm.save_figure(fig)

plt.show()
