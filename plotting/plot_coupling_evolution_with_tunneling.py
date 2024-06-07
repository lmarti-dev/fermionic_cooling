import matplotlib.pyplot as plt
from openfermion import get_sparse_operator
import os
import io
import numpy as np

from data_manager import ExperimentDataManager
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from fauplotstyle.styler import use_style


def main():
    dry_run = True
    edm = ExperimentDataManager(
        "plotting_tunn_vs_coulomb_fh_32",
        tags="plotting",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    ledm = ExperimentDataManager.load_experiment_manager(
        experiment_dirname=r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_06_07\t_probe_couplers_fh_slater"
    )
    all_overlaps = []
    ts = []
    for r in range(ledm.run_number + 1):

        items = ledm.saved_dicts(r)
        d = ledm.load_saved_dict(dict_filename=items[0], run_number=r)
        print(d["max_gs_overlap"])
        all_overlaps.append(np.max(d["overlap_couplers"]))
        ts.append(d["t"])

    all_overlaps = np.array(all_overlaps)
    # t_overlaps = np.max(all_overlaps, axis=(1, 2))

    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(ts, all_overlaps)
    ax.set_xscale("log")
    ax.set_xlabel("tunneling/Coulomb")
    ax.set_ylabel(
        r"$max_{i,j,k}(\langle\psi_0|\tilde{\psi}_i\rangle\langle\tilde{\psi}_j|\psi_k\rangle)$"
    )
    plt.show()


if __name__ == "__main__":
    use_style()
    main()
