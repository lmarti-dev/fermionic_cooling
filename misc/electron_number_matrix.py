import numpy as np
from qutlet.utilities import chained_matrix_multiplication, to_bitstring, hamming_weight

z = np.array([[1, 0], [0, -1]])


e = np.eye(2)
n = (e - z) / 2

n_qubits = 5

mats = np.zeros((2**n_qubits, 2**n_qubits, n_qubits))

for nq in range(n_qubits):
    mats[:, :, nq] = chained_matrix_multiplication(
        np.kron, *[*[e for _ in range(nq)], n, *[e for _ in range(n_qubits - nq - 1)]]
    )

di = np.diag(np.sum(mats, axis=2))

for x in range(2**n_qubits):
    bs = to_bitstring(x, n_qubits=n_qubits)
    hw = hamming_weight(bs)
    print(f"hamm: {hw}, dia: {di[x]}")
