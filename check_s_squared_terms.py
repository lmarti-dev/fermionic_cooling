from helpers.specificModel import SpecificModel
from openfermion import normal_ordered, jordan_wigner

n_qubits = 10

p1 = SpecificModel.s_squared_penalty(n_qubits=n_qubits)
p2 = SpecificModel.s_squared_penalty(n_qubits=n_qubits, alternate=True)

p1no = normal_ordered(p1)
p2no = normal_ordered(p2)

print(p1no == p2no)

print("simple terms:", len([x for x in p1]))
print("altern terms:", len([x for x in p2]))


jw_singlet = jordan_wigner(p1no)

print("singlet len: ", len([pstr for pstr in jw_singlet]))
print("singlet max: ", len(max([str(pstr) for pstr in jw_singlet])))

jw_doublet = jordan_wigner((p1no - 1) ** 2)

print("doublet len: ", len([pstr for pstr in jw_doublet]))
print("doublet max: ", len(max([str(pstr) for pstr in jw_doublet])))
