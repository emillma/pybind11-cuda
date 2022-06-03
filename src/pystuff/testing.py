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


n = 50000000
np.random.seed(123)
message = np.random.randint(0, 1, (n,), np.uint8)


def foo(data):
    for i in range(1000):
        a = 1
    return


functions = [
    foo,
    # pycrc.crc32mpeg2,
    # pycrc.crc32mpeg2_lookup,
    # pycrc.crc32mpeg2_jited,
    pycrc.crc32mpeg2_lookup_jited,
    mycrclib.get_crc,
    mycrclib.get_crc_lookup,
    mycrclib.get_crc_parallel,
    mycrclib.get_crc_lookup_parallel
]

results = []
for func in functions:
    output, time = timeit(func, message, times=1)
    name = str(func)
    print(f"{name: <70}: {output}, {time}")
    results.append((name, output, time))


# for (name, out, t) in results:

# x = cp.ones(1, np.float32)
# y = cp.ones(1, np.float32)

# nbarray = cuda.to_device(np.ones(3, dtype=np.float32))


# a = mycrclib.pyadd(len(x), x.data.ptr, y.data.ptr)
# a = mycrclib.pyadd(len(x), x.data.ptr,
#                    nbarray.__cuda_array_interface__['data'][0])
# print(type(a))
