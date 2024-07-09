import cirq
import numpy as np

qs = cirq.LineQubit.range(1)

ancillas = sum([cirq.Z(q) for q in qs])
print(ancillas)
ancillas = sum(
    [
        x * pstring
        for x, pstring in zip([2.0**-pw for pw in range(len(ancillas))], ancillas)
    ]
)
print(ancillas)
