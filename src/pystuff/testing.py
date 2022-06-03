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


n = 1024*800
# n = n - n % 32
# np.random.seed(123)
message = np.random.randint(0, 256, (n,), np.uint8)
message[1] = 0

crcfull = pycrc.crc32mpeg2_lookup_jited(message)
crca = pycrc.crc32mpeg2_lookup_jited(message[:n//2])
crcb = pycrc.crc32mpeg2_lookup_jited(message[n//2:])
crc_joined = pycrc.join_lookup(crca, crcb, n//2)

print(pycrc.gen_table(1024)[2, 2])
functions = [
    # pycrc.crc32mpeg2,
    # pycrc.crc32mpeg2_lookup,
    pycrc.crc32mpeg2_jited,
    pycrc.crc32mpeg2_lookup_jited,
    mycrclib.get_crc,
    mycrclib.get_crc_lookup,
    mycrclib.get_crc_lookup_parallel,
]

results = []
for func in functions:
    output, time = timeit(func, message, times=1)
    name = str(func)
    print(f"{name: <80}: {output: <12}, {time}")
    results.append((name, output, time))

d_message = cp.asarray(message)
functions = [
    # mycrclib.get_crc_cuda
]

results = []
for func in functions:
    output, time = timeit(func,
                          d_message.data.ptr, n,
                          times=1)
    name = str(func)
    print(f"{name: <80}: {output}, {time}")
    results.append((name, output, time))
