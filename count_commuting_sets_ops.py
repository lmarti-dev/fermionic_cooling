from helpers.specificModel import SpecificModel
from cirq import commutes
from fauvqe.utilities import flatten
from openfermion import fermi_hubbard, jordan_wigner, qubit_operator_to_pauli_sum
from cirq import LineQubit, PauliSum
import io

import json
from tqdm import tqdm
import multiprocessing as mp


def pool_computes(n_jobs, model_names):
    process_pool = mp.Pool(n_jobs)
    results = process_pool.map(commuting_spm_set, model_names)
    # should the process pool be in init and kept throughout the instance life?
    process_pool.close()
    process_pool.join()
    return results, model_names


def get_commuting_sets(pauli_sum: PauliSum):
    n_terms = len(pauli_sum)
    pauli_list = [pstr for pstr in pauli_sum]
    commuting_sets = []
    for ind1 in tqdm(range(n_terms)):
        commuting_set = [
            ind1,
        ]
        second_range = [
            ind2
            for ind2 in range(ind1 + 1, n_terms)
            if not ind2 in flatten(commuting_sets)
        ]
        for ind2 in second_range:
            # print(commuting_set)
            # print(ind1, ind2, "commute!")
            if all(
                commutes(pauli_list[ind3], pauli_list[ind2]) for ind3 in commuting_set
            ):
                commuting_set.append(ind2)
            else:
                pass
                # print(ind1, ind2, "don't commute :(")

        if not any(set(commuting_set) <= set(cset) for cset in commuting_sets):
            commuting_sets.append(commuting_set)
            # print(len(commuting_sets))
    return commuting_sets


def commuting_spm_set(model_name):
    spm = SpecificModel(model_name=model_name)
    hamiltonian = spm.current_model.hamiltonian
    commuting_set = get_commuting_sets(hamiltonian)

    print(model_name, len(commuting_set))
    return commuting_set


def get_molecules_commuting_sets():
    available_models = SpecificModel.get_available_models()
    commuting_sets_dict = {}
    results, model_names = pool_computes(n_jobs=32, model_names=available_models)
    for result, model_name in zip(results, model_names):
        commuting_sets_dict[model_name] = len(result)
    print(commuting_sets_dict.items())
    wout = io.open("commuting_sets.txt", "w+", encoding="utf8")
    wout.write(json.dumps(commuting_sets_dict, ensure_ascii=False, indent=4))
    wout.close()


def get_commuting_fh():
    xx = 2
    yy = 2
    qubits = LineQubit.range(xx * yy * 2)
    fh = fermi_hubbard(2, 2, 1, 2)
    fh_jw = jordan_wigner(fh)
    psum = qubit_operator_to_pauli_sum(operator=fh_jw, qubits=qubits)
    commuting_set = get_commuting_sets(pauli_sum=psum)
    print("fermi hubbard", len(commuting_set))


if __name__ == "__main__":
    get_molecules_commuting_sets()
