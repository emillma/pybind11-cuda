import numpy as np
import sys
from pathlib import Path
import cupy as cp
import numba as ba
from numba import cuda
import crc as pycrc
from timing import timeit

if True:
    build_dir = Path(__file__).parents[2].joinpath('build')
    sys.path.append(str(build_dir.joinpath('src/crclib')))
    import mycrclib


emil = 128000
n = 320*emil*4
# n = n - n % 32

# np.random.seed(123)
message = np.random.randint(0, 256, (n,), np.uint8)


args_norm = [message]

parallel_table = pycrc.get_crcjoin_table(n//16)
args_par = [message, parallel_table]

d_table = cp.asarray(pycrc.get_table_0())
d_message = cp.asarray(message.view(np.uint32))
size = d_message.size
d_result = cp.zeros((10000), np.uint32)
args_cuda = [d_message.data.ptr, size, d_table.data.ptr, d_result.data.ptr]

functions = [
    # [pycrc.crc32,                       norm_args],
    # [pycrc.crc32_lookup,                norm_args],
    [pycrc.crc32_jit,                   args_norm],
    [pycrc.crc32_lookup_jit,            args_norm],
    [mycrclib.get_crc,                  args_norm],
    [mycrclib.get_crc_lookup,           args_norm],

    [pycrc.crc32_parallel,              args_par],
    [mycrclib.get_crc_lookup_parallel,  args_par],

    [mycrclib.get_crc_cuda,             args_cuda],
]

results = []
for (func, args) in functions:
    output, time = timeit(func, *args, times=1)
    name = str(func)
    print(f"{name: <80}: {output: <12}, {time}")
    results.append((name, output, time))

tmp = d_message.reshape(320, -1)

print(pycrc.crc32_jit(message[:emil*4]))
print(d_result[0])
k = 1
print(pycrc.crc32_jit(message[k*emil*4:emil*(k+1)*4]))
print(d_result[k])


a = 0
