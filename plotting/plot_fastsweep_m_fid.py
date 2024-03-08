from json_extender import ExtendedJSONDecoder
from data_manager import ExperimentDataManager
import json
import os
import matplotlib.pyplot as plt
from fauplotstyle.styler import use_style
import numpy as np
import io


def load_json(fpath):
    return json.loads(
        io.open(fpath, "r", encoding="utf8").read(), cls=ExtendedJSONDecoder
    )


def get_times_from_comp(dirname):
    files = os.listdir(dirname)
    times = np.zeros((len(files) // 2 + 1, 2))
    for f in files:
        jobj = load_json(os.path.join(dirname, f))
        if "none" in f:
            col = 1
        else:
            col = 0
        times[jobj["n_gaps"], col] = jobj["total_cool_time"][0][-1]
        times[0, col] = jobj["total_sweep_time"][0][0]
        times[0, col] = jobj["total_sweep_time"][0][0]
    return times


def plot_fast_sweep_vs_m(dirname):
    files = os.listdir(dirname)
    final_fids = np.zeros((len(files) // 2 + 1, 2))
    for f in files:
        jobj = load_json(os.path.join(dirname, f))
        if "none" in f:
            col = 1
        else:
            col = 0
        final_fids[jobj["n_gaps"], col] = jobj["fidelities"][0][-1]
        final_fids[0, col] = jobj["fidelities"][0][0]
        final_fids[0, col] = jobj["fidelities"][0][0]

    fig, ax = plt.subplots()
    ms = np.arange(0, len(final_fids[:, 0]))
    ax.plot(
        ms,
        final_fids[:, 0],
        marker="x",
        label="With fast sweep",
    )
    ax.plot(
        ms,
        final_fids[:, 1],
        marker="d",
        label="Without fast sweep",
    )

    ax.set_ylabel("Final fidelity")
    ax.set_xlabel(r"$M$")

    ax.legend()
    return fig


if __name__ == "__main__":
    use_style()
    edm = ExperimentDataManager(experiment_name="plot_fastsweep_vs_m", dry_run=False)

    which = "coulomb"
    if which == "slater":
        dirname = r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_10h10\run_00000\data"
    elif which == "coulomb":
        dirname = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_08h57\run_00000\data"
        dirname = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\compare_fast_sweep_coulomb_index_2_bad\run_00000\data"
        dirname = r"c:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_11h48\run_00000\data"
        dirname = r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_17h20\run_00000\data"

    fig = plot_fast_sweep_vs_m(dirname)
    edm.save_figure(fig)
    plt.show()