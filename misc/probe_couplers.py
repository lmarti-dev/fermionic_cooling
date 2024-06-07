import matplotlib.pyplot as plt
from openfermion import get_sparse_operator
import numpy as np

from chemical_models.specificModel import SpecificModel
from data_manager import ExperimentDataManager
from fauvqe.models import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number
from fauplotstyle.styler import use_style


def __main__():

    # model_name = "cooked/Fe3_NTA_doublet_7e_12q"
    model_name = "fh_slater"

    model_savename = model_name.replace("/", "_")

    dry_run = False
    edm = ExperimentDataManager(
        f"t_probe_couplers_{model_savename}",
        notes="doing a more thorough investigation of tunneling vs coulomb",
        project="fermionic cooling",
        tags="couplers,overlap,probing stuff",
        dry_run=dry_run,
    )

    new_run = False
    for tpow in np.linspace(-1, 1, 30):
        t = 10**tpow
        if new_run:
            edm.new_run(f"t={t}")
        new_run = True
        if "fh_" in model_name:
            model = FermiHubbardModel(
                x_dimension=4, y_dimension=2, tunneling=t, coulomb=1
            )
            edm.var_dump(model=model.to_json_dict()["constructor_params"])
            n_qubits = len(model.flattened_qubits)
            n_electrons = [4, 4]
            if "coulomb" in model_name:
                start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
                couplers_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
            elif "slater" in model_name:
                start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
                couplers_fock_hamiltonian = start_fock_hamiltonian
        else:
            spm = SpecificModel(model_name=model_name)
            model = spm.current_model
            n_qubits = len(model.flattened_qubits)
            n_electrons = spm.Nf
            start_fock_hamiltonian = spm.current_model.quadratic_terms
            couplers_fock_hamiltonian = start_fock_hamiltonian

        _, sys_eig_states = jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                model.fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )
        _, couplers_sys_eig_states = jw_eigenspectrum_at_particle_number(
            sparse_operator=get_sparse_operator(
                couplers_fock_hamiltonian,
                n_qubits=len(model.flattened_qubits),
            ),
            particle_number=n_electrons,
            expanded=False,
        )

        overlap_gs = np.zeros((couplers_sys_eig_states.shape[-1],))
        overlap_couplers = np.zeros(
            (
                couplers_sys_eig_states.shape[-1] - 1,
                sys_eig_states.shape[-1] - 1,
            )
        )
        for ind_a, coupler_ket in enumerate(couplers_sys_eig_states.T):
            left_term = np.dot(np.conj(sys_eig_states[:, 0]), coupler_ket)
            overlap_gs[ind_a] = np.abs(left_term)

        max_left_term_idx = np.argmax(overlap_gs)
        print(
            f"HIGHEST IDX OVERLAP SLATER/GS: {max_left_term_idx}, value: {np.abs(overlap_gs[max_left_term_idx])}"
        )
        for ind_b in range(1, couplers_sys_eig_states.shape[1]):
            for ind_c in range(1, sys_eig_states.shape[1]):
                right_term = np.dot(
                    np.conj(couplers_sys_eig_states[:, ind_b]), sys_eig_states[:, ind_c]
                )
                overlap_couplers[ind_b - 1, ind_c - 1] = np.abs(
                    overlap_gs[max_left_term_idx] * right_term
                )

        edm.save_dict(
            {
                "overlap_couplers": overlap_couplers,
                "overlap_gs": overlap_gs,
                "max_gs_overlap_idx": int(max_left_term_idx),
                "max_gs_overlap": overlap_gs[max_left_term_idx],
                "t": t,
            }
        )

        fig, ax = plt.subplots()
        ax: plt.Axes
        ax.plot(
            np.arange(overlap_couplers.shape[1]) + 1,
            np.max(overlap_couplers, axis=0),
            "rx:",
            label="max over j",
            linewidth=0.5,
        )
        ax.plot(
            np.arange(overlap_couplers.shape[1]) + 1,
            np.mean(overlap_couplers, axis=0),
            "gd:",
            label="mean over j",
            linewidth=0.5,
        )
        ax.plot(
            np.arange(overlap_couplers.shape[1]) + 1,
            np.diag(overlap_couplers),
            "ms:",
            label=r"$\langle \tilde{\psi}_j | \psi_j \rangle$",
            linewidth=0.5,
        )
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(overlap_couplers.shape[1]) + 1,
            np.argmax(overlap_couplers, axis=0),
            "bo:",
            label="argmax over j",
            linewidth=0.5,
        )

        ax2.plot(
            np.arange(overlap_couplers.shape[1]),
            np.arange(overlap_couplers.shape[1]),
            "k",
            label=r"$j=k$",
            linewidth=0.5,
        )
        ax.set_xlabel(
            r"$k : \langle \psi_0 | \tilde{\psi}_{\mu} \rangle\langle \tilde{\psi}_j | \psi_k \rangle, \ \mu = argmax_k(\psi_0 | \tilde{\psi}_{k})$"
        )
        ax.set_ylabel(
            r"$\langle \psi_0 | \tilde{\psi}_{\mu} \rangle\langle \tilde{\psi}_j | \psi_k \rangle, \ \mu = argmax_k(\psi_0 | \tilde{\psi}_{k})$"
        )
        ax2.set_ylabel(
            r"$j: \langle \psi_0 | \tilde{\psi}_{\mu} \rangle\langle \tilde{\psi}_j | \psi_k \rangle, \ \mu = argmax_k(\psi_0 | \tilde{\psi}_{k})$"
        )

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        edm.var_dump(model_name=model_name, n_qubits=n_qubits, n_electrons=n_electrons)
        edm.save_figure(fig=fig, fig_shape="double-wide")
        # plt.show()


if __name__ == "__main__":
    use_style()
    __main__()
