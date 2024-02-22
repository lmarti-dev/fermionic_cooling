import matplotlib.pyplot as plt
import io
from data_manager import ExtendedJSONDecoder, ExperimentDataManager
import os
import json
import numpy as np

from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from openfermion import get_sparse_operator
from fauvqe.utilities import jw_get_true_ground_state_at_particle_number
from fauplotstyle.styler import use_style


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
    jobj = json.loads(
        io.open(os.path.join(data_dir, files[0]), "r", encoding="utf8").read(),
        cls=ExtendedJSONDecoder,
    )
    return jobj


def plot_separate(edm: ExperimentDataManager, dirname):

    runs = os.listdir(dirname)

    initial_betas = []
    target_betas = []
    final_fids = []
    initial_fids = []

    for run in runs:
        data = get_data(os.path.join(dirname, run, "data"))
        var_dump = get_var_dump(os.path.join(dirname, run, "logging"))
        initial_betas.append(var_dump["initial_beta"])
        target_betas.append(var_dump["target_beta"])
        final_fids.append(data["fidelities"][-1])
        initial_fids.append(data["fidelities"][0])

    fig, ax = plt.subplots()

    if len(list(set(initial_betas))) == 1:
        ax.plot(target_betas, final_fids, label="Final fidelity")
        ax.plot(target_betas, initial_fids, label="Initial fidelity")
        ax.set_xlabel(r"Target $\beta$")
    else:
        ax.plot(initial_betas, final_fids, label="Final fidelity")
        ax.plot(initial_betas, initial_fids, label="Initial fidelity")
        ax.set_xlabel(r"Initial $\beta$")

    ax.set_ylabel(r"Fidelity")
    ax.set_xscale("log")
    ax.legend()
    # fig.suptitle(
    #     rf"1 $\times$ 2 Fermi-Hubbard from $\beta$ to ${target_betas[0]/initial_betas[0]:.1f}\cdot \beta$ "
    # )
    # fig.suptitle(rf"1 $\times$ 2 Fermi-Hubbard from ${initial_betas[0]}$ to $\beta$ ")

    plt.show()
    edm.save_figure(fig)


def plot_single(edm: ExperimentDataManager, dirnames):

    model = FermiHubbardModel(x_dimension=1, y_dimension=2, tunneling=1, coulomb=2)
    n_electrons = [1, 1]
    n_qubits = len(model.flattened_qubits)

    ground_energy, _ = jw_get_true_ground_state_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
    )

    fig, ax = plt.subplots()
    markers = iter("oxsd")
    for dirname in dirnames:
        runs = os.listdir(dirname)

        initial_betas = []
        target_betas = []
        final_fids = []
        initial_fids = []

        for run in runs:
            data = get_data(os.path.join(dirname, run, "data"))
            var_dump = get_var_dump(os.path.join(dirname, run, "logging"))
            initial_betas.append(var_dump["initial_beta"])
            target_betas.append(var_dump["target_beta"])
            final_fids.append(data["fidelities"][-1])
            initial_fids.append(data["fidelities"][0])
        target_betas = np.array(target_betas) * np.abs(ground_energy)
        ax.plot(
            target_betas,
            np.array(final_fids),
            label=rf"$\beta_{{initial}}={initial_betas[0]}$ ",
            marker=next(markers),
        )
    ax.set_xlabel(r"$\beta_{target} E_0$")

    ax.set_ylabel(r"Final fidelity")
    ax.set_xscale("log")
    ax.legend()
    # fig.suptitle(
    #     rf"1 $\times$ 2 Fermi-Hubbard from $\beta$ to ${target_betas[0]/initial_betas[0]:.1f}\cdot \beta$ "
    # )
    # fig.suptitle(rf"1 $\times$ 2 Fermi-Hubbard from ${initial_betas[0]}$ to $\beta$ ")
    edm.save_figure(fig, filename="consolidated_thermal_fig")

    plt.show()


def old_plot():
    # weakencoupling at 10 and add_cross_coupling False 2env qubits
    dirnames = (
        "c:/Users/Moi4/Desktop/current/FAU/phd/data/2024_02_15/compare_thermalizers_11h50",
        "c:/Users/Moi4/Desktop/current/FAU/phd/data/2024_02_15/compare_thermalizers_11h48",
        "c:/Users/Moi4/Desktop/current/FAU/phd/data/2024_02_15/compare_thermalizers_13h50",
        "c:/Users/Moi4/Desktop/current/FAU/phd/data/2024_02_15/compare_thermalizers_15h27",
    )
    labels = (
        r"$10\cdot\beta$ to $\beta$",
        r"$.1\cdot\beta$ to $\beta$",
        r"0 to $\beta$",
        r"100 to $\beta$",
    )
    return dirnames, labels


def newer_plot():
    # weaken coupling set to 1000 and add_cross_coupling True 2 env qubits
    dirnames = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_0_11h07",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_1_11h09",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_100_11h12",
    )
    labels = (
        r"0 to $\beta$",
        r"1 to $\beta$",
        r"100 to $\beta$",
    )
    return dirnames, labels


def newnew_plot():
    # weaken coupling set to 100 and add_cross_coupling True and 10 reps 2 env qubits
    dirnames = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_0_11h20",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_1_11h40",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_100_12h01",
    )
    labels = (
        r"0 to $\beta$",
        r"1 to $\beta$",
        r"100 to $\beta$",
    )
    return dirnames, labels


def afternoon_plot():
    # weaken coupling set to 100 and add_cross_coupling True and 10 reps 1 env qubits
    dirnames = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_0_14h33",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_1_14h35",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_20\compare_thermalizers_initb_10_17h19",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_100_14h36",
    )
    return dirnames


def evening_plot():
    # weaken coupling set to 1 and add_cross_coupling True and 10 reps 1 env qubits
    dirnames = (
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_0_15h40",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_1_15h44",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_20\compare_thermalizers_initb_10_17h13",
        r"c:\Users\Moi4\Desktop\current\FAU\phd\data\2024_02_16\compare_thermalizers_initb_100_15h48",
    )
    labels = (
        r"0 to $\beta$",
        r"1 to $\beta$",
        r"10 to $\beta$",
        r"100 to $\beta$",
    )
    return dirnames, labels


if __name__ == "__main__":
    use_style()
    dirnames = afternoon_plot()

    dry_run = False
    edm = ExperimentDataManager(
        experiment_name="plot_comparison_thermalizing", dry_run=dry_run
    )
    plot_single(edm, dirnames)
