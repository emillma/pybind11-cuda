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


n = 1024*100
# n = n - n % 32
# np.random.seed(123)
message = np.random.randint(0, 256, (n,), np.uint8)

table = pycrc.gen_table(n//8)
norm_args = [message]
par_args = [message, table]

print(pycrc.gen_table(1024)[2, 2])
functions = [
    [pycrc.crc32mpeg2,                norm_args],
    [pycrc.crc32mpeg2_lookup,           norm_args],
    [pycrc.crc32mpeg2_jited,            norm_args],
    [pycrc.crc32mpeg2_lookup_jited,     norm_args],
    [mycrclib.get_crc,                  norm_args],
    [mycrclib.get_crc_lookup,           norm_args],
    [mycrclib.get_crc_lookup_parallel,  par_args],
    [pycrc.crc32mpeg2_parallel_jited,   par_args],
]

results = []
for (func, args) in functions:
    output, time = timeit(func, *args, times=1)
    name = str(func)
    print(f"{name: <80}: {output: <12}, {time}")
    results.append((name, output, time))
