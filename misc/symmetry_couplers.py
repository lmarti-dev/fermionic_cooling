from fermionic_cooling.utils import s_squared_penalty
from openfermion import jordan_wigner, normal_ordered

n_qubits = 8

n_electrons = [2, 2]
penalty = jordan_wigner(s_squared_penalty(n_qubits=n_qubits, n_electrons=n_electrons))


print(penalty)
