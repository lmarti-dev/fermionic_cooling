import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from adiabatic_sweep import (
    fermion_to_dense,
    get_instantaneous_ground_states,
    run_sweep,
    get_sweep_norms,
)
from cirq import fidelity
from json_extender import ExtendedJSONDecoder
from openfermion import (
    get_quadratic_hamiltonian,
    get_sparse_operator,
)

from fermionic_cooling.utils import get_min_gap
from chemical_models.specificModel import SpecificModel
from data_manager import ExperimentDataManager
from fauvqe.models.fermiHubbardModel import FermiHubbardModel
from fauvqe.utilities import jw_eigenspectrum_at_particle_number


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
        times[jobj["n_gaps"], col] = jobj["total_cool_time"]
        times[0, col] = jobj["total_sweep_time"]
        times[0, col] = jobj["total_sweep_time"]
    return np.mean(times, axis=1)


def plot_fidelity(fidelities, instant_fidelities):
    fig, ax = plt.subplots()

    ax.plot(range(len(fidelities)), fidelities, label="g.s.")
    ax.plot(range(len(instant_fidelities)), instant_fidelities, label="instant. g.s.")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Fidelity")
    ax.legend()

    # plt.show()


def chemicals():
    model_names = (
        "v3/FAU_O2_singlet_6e_4o_CASSCF",
        "v3/FAU_O2_singlet_8e_6o_CASSCF",
        "v3/Fe3_NTA_quartet_CASSCF",
        "v3/FAU_O2_triplet_6e_4o_CASSCF",
    )
    dry_run = False

    edm = ExperimentDataManager(
        experiment_name="adiabatic_sweep_molecules",
        notes="adiabatic sweep for various chemicals",
        dry_run=dry_run,
    )
    for model_name in model_names:
        run_comp(edm, model_name)
        edm.new_run()


def coulomb_start():

    dry_run = False
    model_name = "fh_coulomb"

    edm = ExperimentDataManager(
        experiment_name="adiabatic_coulomb_model",
        notes="adiabatic sweep for from the t=0 model",
        dry_run=dry_run,
    )
    run_comp(edm, model_name)


def run_comp(edm: ExperimentDataManager, model_name: str):
    # whether we want to skip all saving data

    if "slater" in model_name:
        dirname = r"C:\Users\Moi4\Desktop\current\FAU\phd\projects\cooling_fermions\graph_data\cooling_with_initial_adiab_sweep_10h10\run_00000\data"
        total_times = get_times_from_comp(dirname)
        total_time = np.max(total_times)
    elif "coulomb" in model_name:
        dirname = None
        total_time = 1e5

    # model stuff
    if "fh_" in model_name:
        model = FermiHubbardModel(x_dimension=2, y_dimension=2, tunneling=1, coulomb=2)
        n_qubits = len(model.flattened_qubits)
        n_electrons = [2, 2]
        if "coulomb" in model_name:
            start_fock_hamiltonian = model.coulomb_model.fock_hamiltonian
        elif "slater" in model_name:
            start_fock_hamiltonian = model.non_interacting_model.fock_hamiltonian
    else:
        spm = SpecificModel(model_name=model_name)
        model = spm.current_model
        n_qubits = len(model.flattened_qubits)
        n_electrons = spm.Nf
        start_fock_hamiltonian = get_quadratic_hamiltonian(
            fermion_operator=model.fock_hamiltonian,
            n_qubits=n_qubits,
            ignore_incompatible_terms=True,
        )

    sys_eig_energies, sys_eig_states = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(model.fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    start_eigenenergies, start_eigenstates = jw_eigenspectrum_at_particle_number(
        sparse_operator=get_sparse_operator(start_fock_hamiltonian),
        particle_number=n_electrons,
        expanded=True,
    )

    gs_index = 0

    final_ground_state = sys_eig_states[:, 0]
    initial_ground_state = start_eigenstates[:, gs_index]
    print(
        f"initial fidelity: {fidelity(initial_ground_state,final_ground_state,qid_shape=(2,)*n_qubits)}"
    )

    ham_start = fermion_to_dense(start_fock_hamiltonian)
    ham_stop = fermion_to_dense(model.fock_hamiltonian)

    # total steps
    # total time
    spectrum_width = np.max(sys_eig_energies) - np.min(sys_eig_energies)

    epsilon = 1e-2
    min_gap = get_min_gap(sys_eig_energies, 1e-2)

    maxh, maxhd = get_sweep_norms(ham_start=ham_start, ham_stop=ham_stop)

    print(f"min gap {min_gap} maxh: {maxh} maxhd: {maxhd}")

    total_time = maxhd**2 * spectrum_width / (min_gap**3 * epsilon)
    n_steps = int(total_time**3 * min_gap**2 * 3 * maxh**2 / (maxhd**2))

    total_time = 1000
    n_steps = 10000

    use_inst_gs = False
    if use_inst_gs:
        instantaneous_ground_states = get_instantaneous_ground_states(
            ham_start=ham_start,
            ham_stop=ham_stop,
            n_steps=n_steps,
            n_electrons=n_electrons,
        )
    else:
        instantaneous_ground_states = None

    print(f"Simulating for {total_time} time and {n_steps} steps")
    (
        fidelities,
        instant_fidelities,
        final_ground_state,
        final_state,
    ) = run_sweep(
        initial_state=initial_ground_state,
        ham_start=ham_start,
        ham_stop=ham_stop,
        final_ground_state=final_ground_state,
        instantaneous_ground_states=instantaneous_ground_states,
        n_steps=n_steps,
        total_time=total_time,
        get_populations=False,
    )

    edm.dump_some_variables(
        model_name=model_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        sys_eig_energies=sys_eig_energies,
        model=model.to_json_dict()["constructor_params"],
    )

    edm.save_dict_to_experiment(
        {
            "times": np.linspace(0, total_time, len(fidelities)),
            "fidelities": fidelities,
            "model_name": model_name,
            "gs_index": gs_index,
        }
    )


def __main__():
    edm = ExperimentDataManager(experiment_name="loong_adiabatic_sweeps")
    run_comp(edm, model_name="fh_coulomb")


if __name__ == "__main__":
    __main__()
