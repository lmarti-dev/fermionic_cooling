import helpers.running_tools as rtools
import helpers.hamiltonian_loading as hamlo

import os
import numpy as np
import helpers.plotting_tools as ptools
import helpers.qubit_tools as qtools
import openfermion as of
import cirq
from helpers.specificModel import SpecificModel
from scipy.sparse.linalg import eigsh

from fauvqe.utilities import depth, jw_eigenspectrum_at_particle_number, qmap


def eigenspectrum(model_name, k=None):
    spm = SpecificModel(model_name=model_name)
    cfm = spm.current_model
    sparse_operator = of.get_sparse_operator(cfm.fock_hamiltonian)

    if k is not None:
        sparse = True
    else:
        sparse = False
    eigvals, eigvecs = jw_eigenspectrum_at_particle_number(
        sparse_operator=sparse_operator,
        particle_number=cfm.Nf,
        expanded=True,
        sparse=sparse,
        k=k,
    )
    return eigvals, eigvecs


def real_expectation(fop, n_qubits, wf):
    return np.real(of.expectation(of.get_sparse_operator(fop, n_qubits), wf))


def compute_number_expectations(model, wf, n_qubits):
    n_up_op, n_down_op, n_total_op = model.hamiltonian_spin_and_number_operator()
    up_spins = real_expectation(n_up_op, n_qubits, wf)
    down_spins = real_expectation(n_down_op, n_qubits, wf)
    Nf_exp = real_expectation(n_total_op, n_qubits, wf)
    return up_spins, down_spins, Nf_exp


def compute_ground_state(
    model_name, bitstrings: bool = True, overall: bool = False, restricted: bool = False
):
    print("model name: ", model_name)
    spm = SpecificModel(model_name=model_name, restricted=restricted)
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

    print("exact gse: {}".format(exact_gse))
    print("diago gse: {}".format(true_energy))
    print("expec gse: {}".format(expectation))
    print("hfexp ene: {}".format(hf_energy))
    print("hf-gs fid:", rtools.fidelity(true_wf, initial_wf))
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


def run_optim(model_name):
    objective_function = "infidelity"
    objective_options = {
        "objective_function": objective_function,
    }
    optimise_options = {
        "optimise_function": "scipy",
        "method": "Powell",
        "initial_params": "zeros",
        "initial_state": None,
        "maxiter": 1e4,
        "maxfev": 1e3,
        "disp": True,
        "ftol": 1e-13,
    }
    # optimise_options = {
    #     "optimise_function": "adam",
    #     "method": "ADAM",
    #     "break_param": 500,
    #     "eta": 0.001,
    #     "b_1": 0.9,
    #     "b_2": 0.999,
    #     "initial_params": "zeros",
    #     "use_progress_bar": True,
    # }

    ansaetze = [
        # "hva",
        # "brickwall",
        # "pyramid",
        # "totally_connected",
        # "totally_connected_spinc",
        # "stair",
        # "adapt_vqe_paulistr",
        "adapt_vqe_hamiltonian",
        "adapt_vqe_fermionic",
    ]

    layers = 4

    for ansatz in ansaetze:
        spm = SpecificModel(model_name=model_name)
        cfm = spm.current_model
        Nf = spm.Nf
        initial_wf = spm.initial_wf
        exact_gse = spm.ground_state_energy

        true_energy, true_wf = qtools.get_fermionic_model_ground_state(
            model=cfm, Nf=Nf, sparse=False
        )
        objective_options["target_state"] = true_wf
        print("begin optim")
        print("ansatz: {}".format(ansatz))
        model, (wfs, energies, fidelities) = rtools.run_model(
            model=cfm,
            initial_state=None,
            layers=layers,
            ansatz=ansatz,
            optimise_options=optimise_options,
            objective_function=objective_function,
        )
        print("end optim")
        start_wf = wfs[0]
        final_wf = wfs[-1]

        print("start fid:", rtools.fidelity(start_wf, true_wf))
        print("final fid:", rtools.fidelity(true_wf, final_wf))

        datapath = (
            "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/data/basf_quick_results"
        )
        fname = "{ansatz}_{method}_optimisationresult_store.json".format(
            ansatz=ansatz, method=optimise_options["optimise_function"]
        )
        rtools.ensure_fpath(os.path.join(datapath, fname))
        jobj = {
            "model": type(model).__name__,
            "ansatz": ansatz,
            "layers": layers,
            # "objective": objective_options,
            # "optim": optimise_options,
            "wfs": wfs,
            "energies": energies,
            "fidelities": fidelities,
            "depth": depth(model.circuit),
            "params": len(model.circuit_param),
        }
        # print(model.circuit)
        rtools.save_to_json(data=jobj, dirname=datapath, fname=fname)


def save_gs(name):
    true_energy, true_wf = compute_ground_state(name)
    np.save(f"ground_state_{name}_hamiltonian.npy", true_wf)
    np.save(f"ground_energy_{name}_hamiltonian.npy", true_energy)


def save_eigenspectrum(name, k):
    eigvals, eigvecs = eigenspectrum(name, k)
    np.save(f"eigenvalues_{name}_hamiltonian.npy", eigvals)
    np.save(f"eigenstates_{name}_hamiltonian.npy", eigvecs)


def load_ludwig_10_qubits_hf_gs():
    dirname = "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/fauvqe/fauvqe_running_code/sandbox/chemical_hamiltonians/ludwig_10_qubits_model/"
    hf_state = np.load(os.path.join(dirname, "hartreefock.npy"))
    ground_state = np.load(os.path.join(dirname, "exact_eigenstate.npy"))
    return hf_state, ground_state


def load_ludwig_16_qubits_hf_gs():
    dirname = "C:/Users/Moi4/Desktop/current/FAU/phd/code/vqe/fauvqe/fauvqe_running_code/sandbox/chemical_hamiltonians/ludwig_16_qubits_model/"
    hf_state = np.load(os.path.join(dirname, "hartreefock.npy"))
    ground_state = np.load(os.path.join(dirname, "exact_eigenstate.npy"))
    return hf_state, ground_state


def print_comp_wf(wf):
    print(np.argmax(wf))
    print([int(x) for x in wf])


def comparison_lm_lw(bitstrings: bool):
    lw_hf_state, lw_ground_state = load_ludwig_16_qubits_hf_gs()

    spm = SpecificModel(model_name="sixteen")
    cfm = spm.current_model
    Nf = spm.Nf
    lm_hf_state = spm.initial_wf
    exact_gse = spm.ground_state_energy

    lw_hf_energy = cfm.hamiltonian.expectation_from_state_vector(
        lw_hf_state.astype(np.complex64), qubit_map=qmap(model=cfm)
    )
    lw_gs_energy = cfm.hamiltonian.expectation_from_state_vector(
        lw_ground_state.astype(np.complex64), qubit_map=qmap(model=cfm)
    )

    lm_ground_energy, lm_ground_state = compute_ground_state("sixteen")

    print("lw hf expect:", lw_hf_energy)
    print("lw gs expect:", lw_gs_energy)

    print("lw-lm gs fid", rtools.fidelity(lw_ground_state, lm_ground_state))
    print("lw-lm hf fid", rtools.fidelity(lm_hf_state, lw_hf_state))

    if bitstrings:
        print("lw indices")
        SpecificModel.analyse_largest_bitstrings(lw_ground_state)

        print("lm indices")
        SpecificModel.analyse_largest_bitstrings(lm_ground_state)


if __name__ == "__main__":
    # compute_ground_state("ten")
    # compute_ground_state("felowspinavas")
    for model in SpecificModel.get_restricted_models():
        # if model != "ni2_nta":
        compute_ground_state(model, bitstrings=False, restricted=True)
# import matplotlib.pyplot as plt
# cfm, Nf, initial_wf, exact_gse=ten_qubits_model()
# fig = ptools.plot_fock_hamiltonian_representation(cfm.fock_hamiltonian)
# plt.show()
