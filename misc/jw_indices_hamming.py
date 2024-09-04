import numpy as np
from qutlet.utilities import jw_spin_correct_indices, to_bitstring, hamming_weight


n_qubits = 10
n_electrons = [4, 3]

state = np.random.rand(2**n_qubits)

indices = jw_spin_correct_indices(n_electrons=n_electrons, n_qubits=n_qubits)
for idx in sorted(indices):
    bs = to_bitstring(idx, n_qubits)
    nup = hamming_weight(bs[::2])
    ndown = hamming_weight(bs[1::2])
    print(f"{idx}: [{nup},{ndown}], {bs}")
