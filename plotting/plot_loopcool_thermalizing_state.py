from fauplotstyle.styler import use_style
import json
import io
from json_extender import ExtendedJSONDecoder
import matplotlib.pyplot as plt
from data_manager import ExperimentDataManager
import numpy as np


def load_json(fpath):
    return json.loads(
        io.open(fpath, "r", encoding="utf8").read(), cls=ExtendedJSONDecoder
    )


def plot_fid_progression(fids):
    fig, ax = plt.subplots()

    ax.plot(range(len(fids)), 1 - np.array(fids))

    ax.set_xlabel("Iteration step")
    ax.set_ylabel("Infidelity")
    # ax.set_yscale("log")

    return fig


if __name__ == "__main__":
    use_style()

    edm = ExperimentDataManager(experiment_name="single_thermalization_plot")
    fpath = r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_03_21\fh22_0_target_beta_12h15\run_00000\data\thermalizin_free_fh_coulomb_2.json"
    jobj = load_json(fpath)
    fig = plot_fid_progression(jobj["fidelities"])

    edm.var_dump(fpath=fpath)
    edm.save_figure(
        fig, filename="single_therm_zipcool", add_timestamp=False, fig_shape="half-y"
    )
    plt.show()
