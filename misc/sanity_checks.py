import sys
import numpy as np
import cirq
import time

sys.path.append("/home/Refik/Data/My_files/Dropbox/PhD/repos/fauvqe/")
from coolerClass import trace_out_env


def __main__(args):
    ns = 4
    ne = 1
    ds = 2**ns
    de = 2**ne

    rhos = np.random.rand(ds, ds)
    rhos = rhos / np.trace(rhos)
    rhoe = np.random.rand(de, de)
    rhoe = rhoe / np.trace(rhoe)
    rho = np.kron(rhos, rhoe)
    start = time.time()
    rhos_refik = trace_out_env(rho, ns, ne, use_refik=True)
    end = time.time()
    print("Refik time: {}".format(end - start))
    start = time.time()
    rhos_lucas = trace_out_env(rho, ns, ne, use_refik=False)
    end = time.time()
    print("Lucas time: {}".format(end - start))
    start = time.time()
    rhos_cirq = cirq.partial_trace(
        rho.reshape(*[2 for _ in range(2 * ns + 2 * ne)]), range(ns)
    ).reshape(2**ns, 2**ns)
    end = time.time()
    print("Cirq time: {}".format(end - start))
    print("Error Refik: {}".format(np.linalg.norm(rhos - rhos_refik)))
    print("Error Lucas: {}".format(np.linalg.norm(rhos - rhos_lucas)))
    print("Error Cirq: {}".format(np.linalg.norm(rhos - rhos_cirq)))


if __name__ == "__main__":
    __main__(sys.argv)
import sys
