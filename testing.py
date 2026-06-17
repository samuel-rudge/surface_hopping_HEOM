import numpy as np
import sparse_propagation  # the compiled wrapper
from matplotlib import pyplot as plt
import tracemalloc,gc

tracemalloc.start()

def print_memory_diff(snapshot1, snapshot2, msg="Memory difference"):
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print(f"--- {msg} ---")
    for stat in stats[:5]:  # top 5 lines
        print(stat)
    print()

def get_memory_usage_mib(snapshot):
    stats = snapshot.statistics('filename')
    total = sum(stat.size for stat in stats)
    return total / (1024* 1024)

N = 114
npairs = 798
nnz_elements = N

# Fake data
pair_info_row = np.random.randint(0, N, size=npairs).astype(np.int32)
pair_info_col = np.random.randint(0, N, size=npairs).astype(np.int32)
pair_values = np.array(np.random.rand(npairs), dtype=np.float64, order='F')
rho_input = np.random.rand(nnz_elements)
rho_output = np.zeros_like(rho_input)
rho_temp = np.zeros_like(rho_input)
rho_deriv = np.zeros_like(rho_input)
rk_coeff = np.array([1.0, 0.5, 1/6, 1/24])
dt = 0.01
nthreads = 4
max_order = 4

import os

# Set threading configuration BEFORE importing any MKL-using modules
os.environ["MKL_NUM_THREADS"] = str(nthreads_liouvillian)        # Number of MKL threads
# os.environ["OMP_NUM_THREADS"] = "4"        # Number of OpenMP threads
os.environ["MKL_DYNAMIC"] = "FALSE"        # Disable dynamic threading
os.environ["OMP_DYNAMIC"] = "FALSE"        # Disable OpenMP dynamic threading

tracemalloc.start()
snapshot_prev = tracemalloc.take_snapshot()
memory_usage = []
for i in range(500):
    # snapshot_before = tracemalloc.take_snapshot()
    pair_values = np.array(np.random.rand(npairs), dtype=np.float64, order='F')
    rho_output = sparse_propagation.sparse_one_step_propagation(
        pair_info_row=pair_info_row, pair_info_col=pair_info_col, pair_values=pair_values, dt=dt, rho_input=rho_input,
        max_expan_order=max_order, nthreads_liouvillian=nthreads, npairs=npairs, nnz_elements=nnz_elements, rk_coeff=rk_coeff,
        rho_temp=rho_temp, rho_deriv=rho_deriv)
    snapshot_now = tracemalloc.take_snapshot()
    # print_memory_diff(snapshot_prev, snapshot_now, i)
    memory_usage.append(get_memory_usage_mib(snapshot_now))
    snapshot_prev = snapshot_now

tracemalloc.stop()
plt.figure(figsize=(8,5))
plt.plot(memory_usage, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Memory usage (MB)")
plt.title("Memory usage during sparse_one_step_propagation calls")
plt.grid(True)
plt.show()
