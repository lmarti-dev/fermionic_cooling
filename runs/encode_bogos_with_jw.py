from fauvqe.models import FermiHubbardModel
import numpy as np
from openfermion import (
    FermionOperator,
    jordan_wigner,
    hermitian_conjugated,
    normal_ordered,
)
from itertools import combinations
import matplotlib.pyplot as plt


def get_nn_bog_matrix(fock_ham):
    matrix = np.zeros((x * y * 2, x * y * 2))

    for k, v in fock_ham.terms.items():
        coords = (k[0][0], k[1][0])
        matrix[coords] = v
    return matrix + np.conjugate(matrix.T)


def are_bogos_bound_to_zero(bogos):
    ops = []
    for bogo in bogos:
        for term in bogo.terms:
            ops += list(term)
            if len(set(ops)) != len(ops):
                return True
    return False


def remove_null_terms(fop: FermionOperator):
    if fop is None:
        return None
    out_fop = FermionOperator()
    for term in fop.terms:
        if len(set(term)) == len(term):
            out_fop += FermionOperator(term, fop.terms[term])

    return out_fop


def get_bogos(matrix):
    bogos = []
    for m in range(matrix.shape[0]):
        fop = FermionOperator()
        for n in range(matrix.shape[1]):
            fop += FermionOperator(f"{n}^", coefficient=matrix[m, n])
        bogos.append(fop)
    return bogos


def bogoprod(bogos: list, check_bound: bool = True):
    if are_bogos_bound_to_zero(bogos) and check_bound:
        print("bound to zero")
        return None
    fop_out = bogos[0]
    for bogo in bogos[1:]:
        fop_out *= bogo
    return fop_out


def build_bogo_creators(bogos, n_electrons, n_qubits):
    combs = combinations(range(n_qubits), sum(n_electrons))
    bogo_creators = []
    for comb in combs:
        print("combs:", list(comb))
        bogo_list = [bogos[x] for x in comb]
        bogo_creator = bogoprod(bogo_list)
        if bogo_creator is not None:
            bogo_creator = normal_ordered(bogo_creator)
            bogo_creators.append(bogo_creator)
    return bogo_creators


def build_couplers(bogo_creators):
    couplers = []
    pairs = combinations(range(len(bogo_creators)), 2)
    for pair in pairs:
        couplers.append(
            bogo_creators[pair[0]] * hermitian_conjugated(bogo_creators[pair[1]])
        )
    return couplers


def encode_couplers(couplers):
    jw_coupler = []
    for coupler in couplers:
        jw_coupler.append(jordan_wigner(coupler))
    return jw_coupler


def plot_coefficients(jw_couplers):
    total_coefficients = []

    for coupler in jw_couplers:
        # coupler is paulisum
        total_coefficients.append(
            np.array(list(np.real(pstr.coefficient) for pstr in coupler))
        )
    fig, ax = plt.subplots()
    for ind, coeffs in enumerate(total_coefficients):
        ax.plot(
            list(range(len(coeffs))),
            np.sort(np.abs(coeffs))[::-1],
            label=f"$V_k={ind}$",
        )

    ax.set_ylabel("Coefficient")
    ax.set_xlabel("Pauli string index")
    ax.set_yscale("log")
    plt.show()
    return fig


if __name__ == "__main__":

    x = 2
    y = 2
    n_electrons = [2, 2]

    model = FermiHubbardModel(x_dimension=x, y_dimension=y, tunneling=1, coulomb=2)
    fock_ham = model.non_interacting_model.fock_hamiltonian

    matrix = get_nn_bog_matrix(fock_ham)
    eigvals, eigvecs = np.linalg.eigh(matrix)

    print("### getting bogos")
    bogos = get_bogos(eigvecs.T)
    print("n bogos", len(bogos))
    print("bogos", bogos)

    print("### getting b_ib_j builder things")
    bogo_creators = build_bogo_creators(bogos, n_electrons, n_qubits=x * y * 2)
    print("n bogos creators", len(bogo_creators))

    print("### building couplers")
    couplers = build_couplers(bogo_creators)
    print("n couplers", len(couplers))

    print("### jw encoding couplers")
    jw_couplers = encode_couplers(couplers)
    print("n jw couplers", len(jw_couplers))

    print("### plotting")
    fig = plot_coefficients(jw_couplers)
