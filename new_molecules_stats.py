from helpers.specificModel import SpecificModel
import helpers.running_tools as rtools
import helpers.qubit_tools as qtools
from fauvqe.utilities import qmap
import openfermion as of
import numpy as np
from scipy.sparse.linalg import eigsh


def real_expectation(fop, n_qubits, wf):
    return np.real(of.expectation(of.get_sparse_operator(fop, n_qubits), wf))


def compute_number_expectations(model, wf, n_qubits):
    n_up_op, n_down_op, n_total_op = model.hamiltonian_spin_and_number_operator()
    up_spins = real_expectation(n_up_op, n_qubits, wf)
    down_spins = real_expectation(n_down_op, n_qubits, wf)
    Nf_exp = real_expectation(n_total_op, n_qubits, wf)
    return up_spins, down_spins, Nf_exp


def compute_ground_state(model_name, bitstrings: bool = False, overall: bool = False):
    print("=" * 20)
    print("model name: ", model_name)
    spm = SpecificModel(model_name=model_name)
    cfm = spm.current_model

    print("counted qubits: {}".format(of.count_qubits(cfm.fock_hamiltonian)))
    Nf = spm.Nf
    initial_wf = spm.initial_wf
    exact_gse = spm.ground_state_energy

    print("computing true ground wf")
    true_energy, true_wf = qtools.get_fermionic_model_ground_state(
        model=cfm, Nf=Nf, sparse=True
    )
    hf_energy = cfm.hamiltonian.expectation_from_state_vector(
        initial_wf.astype(np.complex64), qubit_map=qmap(model=cfm)
    )
    expectation = cfm.hamiltonian.expectation_from_state_vector(
        true_wf.astype(np.complex64), qubit_map=qmap(model=cfm)
    )

    max_ci_wf = spm.get_max_ci_wf(true_wf)
    max_ci_ene = cfm.hamiltonian.expectation_from_state_vector(
        max_ci_wf.astype(np.complex64), qubit_map=qmap(model=cfm)
    )

    print("exact gse: {}".format(exact_gse))
    print("diago gse: {}".format(true_energy))
    print("expec gse: {}".format(expectation))
    print("hfexp ene: {}".format(hf_energy))
    print("max_ci_ene: {}".format(max_ci_ene))
    print("hf-gs fid:", qtools.fidelity(true_wf, initial_wf) ** 2)
    print("max_ci-gs fid:", qtools.fidelity(true_wf, max_ci_wf) ** 2)
    print("highest gs comp: {}".format(spm.get_largest_n_bitstrings(true_wf, n=1)))
    print("highest hf comp: {}".format(spm.get_largest_n_bitstrings(initial_wf, n=1)))
    n_qubits = spm.n_qubits
    gs_up_spins, gs_down_spins, gs_Nf_exp = compute_number_expectations(
        cfm, true_wf, n_qubits
    )
    hf_up_spins, hf_down_spins, hf_Nf_exp = compute_number_expectations(
        cfm, initial_wf, n_qubits
    )

    print(
        "gs: up spins: {:.3f} down spins: {:.3f} total fermions {:.3f}".format(
            gs_up_spins, gs_down_spins, gs_Nf_exp
        )
    )
    print(
        "hf: up spins: {:.3f} down spins: {:.3f} total fermions {:.3f}".format(
            hf_up_spins, hf_down_spins, hf_Nf_exp
        )
    )

    if bitstrings:
        print("gs indices")
        spm.analyse_largest_bitstrings(true_wf, n=2)
        print("hf indices")
        spm.analyse_largest_bitstrings(initial_wf, n=2)
    if overall:
        print("computing overall ground states")
        print("=" * 10)

        sparse_operator = of.get_sparse_operator(cfm.fock_hamiltonian)
        eigenenergies, eigenstates = eigsh(sparse_operator)
        for i in range(eigenstates.shape[-1]):
            print("{}th lowest eigenstate".format(i + 1))
            print("energy: ", eigenenergies[i])
            spm.analyse_largest_bitstrings(eigenstates[:, i], n=1)

    return true_energy, true_wf


v2_models = SpecificModel.get_model_family("v3")
for model_name in v2_models:
    compute_ground_state(model_name)
