import cirq
from helpers.specificModel import SpecificModel
import helpers.qubit_tools as qtools
import helpers.running_tools as rtools
from qiskit.algorithms.optimizers.nft import nakanishi_fujii_todo
from fauvqe.optimisers.optimisation_result import OptimisationResult
from fauvqe.models.circuits.misc_circuits import totally_connected_ansatz, stair_ansatz


def get_custom_model(model_name) -> SpecificModel:
    spm = SpecificModel(model_name)
    return spm


if __name__ == "__main__":
    # initial_state_names = ("dicke", "hartree_fock", "slater")
    initial_state_name = "Max ci"
    print("=========== COMMENCE {} OPTIMIZATION ===========".format(initial_state_name))

    spm = get_custom_model(model_name="v3/FAU_O2_singlet_6e_4o_CASSCF")
    model = spm.current_model

    # model = get_fh(2, 2, 4)
    model.initial_state_name = initial_state_name

    stair_ansatz(model, layers=30)

    ground_energy, ground_state = qtools.get_fermionic_model_ground_state(
        model, model.Nf
    )

    initial_state = ground_state
    target_state = qtools.get_max_ci_state(ground_state)
    objective_function = "infidelity"
    objective_options = {
        "objective_function": objective_function,
        "target_state": target_state,
    }
    optimise_options = {
        "optimise_function": "scipy",
        "method": nakanishi_fujii_todo,
        "initial_state": initial_state,
        "maxiter": 1e6,
        "maxfev": 1e6,
        "disp": True,
        "ftol": 1e-13,
    }
    init_fid = cirq.fidelity(
        initial_state, target_state, qid_shape=(2,) * len(model.flattened_qubits)
    )
    print("init fid: {}".format(init_fid))
    optimiser, objective, options = rtools.quick_optim_objective_pick(
        model=model,
        objective_options=objective_options,
        optimise_options=optimise_options,
    )

    result: OptimisationResult = optimiser.optimise(
        objective=objective, initial_params="zeros"
    )
    final_state = result.get_latest_step().wavefunction
    final_fid = cirq.fidelity(
        final_state, target_state, qid_shape=(2,) * len(model.flattened_qubits)
    )
    print("final_fid: {}".format(final_fid))

    # print(model.circuit)
