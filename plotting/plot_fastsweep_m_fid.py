from json_extender import ExtendedJSONDecoder
from data_manager import ExperimentDataManager
import json
import os
import matplotlib.pyplot as plt
from fauplotstyle.styler import use_style
import numpy as np
import io
from matplotlib.ticker import MaxNLocator


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


def plot_fast_sweep_vs_m(dirname, max_sweep_fid=None):
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

    if max_sweep_fid is not None:
        ax.hlines(
            max_sweep_fid,
            ms[0],
            ms[-1],
            "r",
            label="Adiabatic sweep",
            linestyles="dashed",
        )

    ax.set_ylabel("Final fidelity")
    ax.set_xlabel(r"$M$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc="center right", bbox_to_anchor=(1, 0.3))
    return fig


if __name__ == "__main__":
    use_style()
    edm = ExperimentDataManager(experiment_name="plot_fastsweep_vs_m", dry_run=True)

    graph_dir = (
        r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data"
    )
    which = "slater"
    if which == "slater":
        dirname = dirname = os.path.join(
            graph_dir, r"cooling_with_initial_adiab_sweep_10h10\run_00000\data"
        )
        dirname = os.path.join(
            graph_dir, r"cooling_with_initial_adiab_sweep_slater_15h12\run_00000\data"
        )
        max_sweep_fid = 0.49844239875687185

    elif which == "coulomb":
        dirname = os.path.join(
            graph_dir, r"cooling_with_initial_adiab_sweep_08h57\run_00000\data"
        )
        dirname = os.path.join(
            graph_dir, r"compare_fast_sweep_coulomb_index_2_bad\run_00000\data"
        )
        dirname = os.path.join(
            graph_dir, r"cooling_with_initial_adiab_sweep_11h48\run_00000\data"
        )
        dirname = os.path.join(
            graph_dir, r"cooling_with_initial_adiab_sweep_17h20\run_00000\data"
        )
        max_sweep_fid = 0.08333327548043287

    fig = plot_fast_sweep_vs_m(dirname, max_sweep_fid=max_sweep_fid)

    edm.dump_some_variables(start_ham=which)
    edm.save_figure(fig, fig_shape="half-y")
    plt.show()
