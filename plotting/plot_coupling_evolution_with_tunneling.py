import matplotlib.pyplot as plt
import numpy as np

from data_manager import ExperimentDataManager
from fauplotstyle.styler import use_style


def main():
    dry_run = False
    edm = ExperimentDataManager(
        "plotting_tunn_vs_coulomb_fh_and_sc",
        tags="plotting",
        notes="plotting the scaling up of t/J overlap (if that makes sense)",
        project="fermionic cooling",
        dry_run=dry_run,
    )
    labels = (
        r"$2 \times 2$",
        r"$2 \times 2$, ts + c",
        r"$3 \times 2$",
        r"$4 \times 2$",
        r"$3 \times 2$, s + c",
        r"$3 \times 2$, ts + c",
    )
    filenames = (
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_06_07\t_probe_couplers_fh_slater",
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_07_01\tvar22_probe_couplers_fh_slater",
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_06_07\t_probe_couplers_fh_slater_00001",
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_06_07\t_probe_couplers_fh_slater_4x2",
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_07_01\t_probe_couplers_fh_slater",
        r"C:\Users\Moi4\Desktop\current\FAU\phd\data\2024_07_01\tvar_probe_couplers_fh_slater",
    )
    edm.var_dump(items={str(s): f for s, f in zip(labels, filenames)})
    for overlap_pool in ("mean", "max"):
        fig, ax = plt.subplots()
        for label, filename in zip(labels, filenames):
            ledm = ExperimentDataManager.load_experiment_manager(
                experiment_dirname=rf"{filename}"
            )
            all_overlaps = []
            ts = []
            for r in range(ledm.run_number + 1):

                items = ledm.saved_dicts(r)
                d = ledm.load_saved_dict(dict_filename=items[0], run_number=r)
                print(d["max_gs_overlap"])

                ts.append(d["t"])

                if overlap_pool == "mean":
                    ax.set_yscale("log")
                    all_overlaps.append(np.mean(d["overlap_couplers"]))
                elif overlap_pool == "max":
                    all_overlaps.append(np.max(d["overlap_couplers"]))

            all_overlaps = np.array(all_overlaps)
            ax: plt.Axes
            ax.plot(ts, all_overlaps, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("tunneling/Coulomb")
            # ax.set_ylabel(
            #     r"$mean_{i,j,k}(\langle\psi_0|\tilde{\psi}_i\rangle\langle\tilde{\psi}_j|\psi_k\rangle)$"
            # )
            ax.set_ylabel(rf"{overlap_pool} transition overlap")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        edm.save_figure(fig, fig_shape="page-wide")


if __name__ == "__main__":

    use_style()
    main()
