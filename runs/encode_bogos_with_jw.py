from qutlet.models import FermiHubbardModel
import numpy as np
from openfermion import (
    FermionOperator,
    jordan_wigner,
    hermitian_conjugated,
    qubit_operator_to_pauli_sum,
)
from itertools import combinations
import matplotlib.pyplot as plt
from fauplotstyle.styler import use_style

from data_manager import ExperimentDataManager
from cirq import LineQubit


def get_nn_bog_matrix(fock_ham):
    matrix = np.zeros((x * y * 2, x * y * 2))

    for k, v in fock_ham.terms.items():
        coords = (k[0][0], k[1][0])
        matrix[coords] = v
    return matrix


def remove_null_terms(fop: FermionOperator):
    if fop is None:
        return None
    out_fop = FermionOperator()
    for term in fop.terms:
        if len(set(term)) == len(term):
            out_fop += FermionOperator(term, fop.terms[term])

    return out_fop


def get_bogos(matrix, spin: int):
    bogos = []
    for m in range(matrix.shape[0]):
        fop = FermionOperator()
        for n in range(matrix.shape[1]):
            fop += FermionOperator(f"{int(2*n + spin)}^", coefficient=matrix[m, n])
        bogos.append(fop)
    return bogos


def bogoprod(bogos: tuple):
    fop_out = FermionOperator.identity()
    for bogo in bogos:
        fop_out *= bogo
        # fop_out = remove_null_terms(fop_out)
    return fop_out


def build_bogo_creators(bogos_up, bogos_down, n_electrons):
    bogo_up_combs = list(combinations(bogos_up, n_electrons[0]))
    bogo_down_combs = list(combinations(bogos_down, n_electrons[1]))
    bogo_creators = []
    for i1, bogo_up_comb in enumerate(bogo_up_combs):
        for i2, bogo_down_comb in enumerate(bogo_down_combs):
            bogo_list = [*bogo_up_comb, *bogo_down_comb]
            bogo_creator = bogoprod(bogo_list)
            if bogo_creator is not None:
                # bogo_creator = normal_ordered(bogo_creator)

                bogo_creators.append(bogo_creator)
            print(i1, i2, len(bogo_creator.terms))

    return bogo_creators


def build_couplers(bogo_creators):
    bogo_0 = bogo_creators[0]
    for bogo_c in bogo_creators[1:]:
        yield bogo_0 * hermitian_conjugated(bogo_c)


def encode_couplers(couplers, qubits):
    for coupler in couplers:
        yield qubit_operator_to_pauli_sum(jordan_wigner(coupler), qubits=qubits)


def pauli_mask_to_pstr(pauli_mask: np.array, qubits):
    d = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    qubits_ind = [q.x for q in qubits]
    sorted_qubs = np.argsort(qubits_ind)

    return "".join(f"{d[pauli_mask[ind]]}_{qubits_ind[ind]}" for ind in sorted_qubs)


def get_coeffs_and_maxpstr(jw_couplers):
    total_coefficients = []
    max_pauli_strs = []
    for coupler in jw_couplers:
        # coupler is paulisum
        pstr_list = list(coupler)
        coefficients = list(np.real(x.coefficient) for x in pstr_list)
        total_coefficients.append(np.array(coefficients))

        max_ind = np.argmax(coefficients)
        max_pstr = pstr_list[max_ind]
        pstr_pretty = pauli_mask_to_pstr(
            max_pstr.dense(qubits=max_pstr.qubits).pauli_mask, max_pstr.qubits
        )
        max_pauli_strs.append(pstr_pretty)
    return total_coefficients, max_pauli_strs


def plot_bogo_jw_coefficients(total_coefficients, max_pauli_strs):

    fig, ax = plt.subplots()
    markers = "xdos1^+"
    cmap = plt.get_cmap("faucmap", len(total_coefficients))
    for ind, coeffs in enumerate(total_coefficients):
        ax.plot(
            list(range(1, len(coeffs) + 1)),
            np.sort(np.abs(coeffs))[::-1],
            label=f"$V_{{({ind+1},0)}}: {max_pauli_strs[ind]}$",
            # marker=markers[ind % len(markers)],
            # markevery=5,
            color=cmap(ind),
        )

    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Pauli string index")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-4, 9e-2])
    ax.set_xlim([1, 3e3])
    # ax.legend(ncol=4)
    return fig


if __name__ == "__main__":

    x = 2
    y = 2
    n_electrons = [2, 2]

    dry_run = False
    edm = ExperimentDataManager(
        experiment_name=f"jw_encode_bogos_coeff_{x}_{y}_{n_electrons[0]}u_{n_electrons[1]}d",
        dry_run=dry_run,
    )

    model = FermiHubbardModel(
        lattice_dimensions=(x, y), n_electrons=n_electrons, tunneling=1, coulomb=2
    )
    fock_ham = model.non_interacting_model.fock_hamiltonian

    matrix = get_nn_bog_matrix(fock_ham)

    sector_matrix = matrix[::2, ::2]
    eigvals, eigvecs = np.linalg.eigh(sector_matrix)

    # eigvecs = np.round(eigvecs, 4)

    edm.var_dump(
        n_electrons=n_electrons,
        matrix=matrix,
        model=model.__to_json__,
    )

    print("### getting bogos")
    bogos_up = get_bogos(eigvecs.T, 0)
    bogos_down = get_bogos(eigvecs.T, 1)

    n_combs = sum(1 for _ in combinations(range(x * y), n_electrons[0])) ** 2

    print(f"there should be {n_combs} creators")

    print("### getting b_ib_j builder things")
    bogo_creators = build_bogo_creators(bogos_up, bogos_down, n_electrons)

    print(f"there are {len(bogo_creators)} creators")
    # print("n bogos creators", len(bogo_creators))

    print("### building couplers")
    print(f"there should be {n_combs} couplers")
    couplers = build_couplers(bogo_creators)
    print("### jw encoding couplers")
    qubits = LineQubit.range(x * y * 2)
    jw_couplers = encode_couplers(couplers, qubits)

    jw_couplers = list(jw_couplers)

    total_coefficients, max_pauli_strs = get_coeffs_and_maxpstr(jw_couplers)

    edm.save_dict(
        {
            "bogos_up": bogos_up,
            "bogos_down": bogos_down,
            "total_coefficients": total_coefficients,
            "max_pauli_strs": max_pauli_strs,
        }
    )

    print("### plotting")

    use_style()

    fig = plot_bogo_jw_coefficients(total_coefficients, max_pauli_strs)

    edm.save_figure(fig, fig_shape="page-wide")
    plt.show()
