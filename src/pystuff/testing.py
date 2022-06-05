import numpy as np
import sys
from pathlib import Path
import cupy as cp
import numba as ba
from numba import cuda
import crc as pycrc
from timing import timeit
from pathlib import Path
if True:
    build_dir = Path(__file__).parents[2].joinpath('build')
    sys.path.append(str(build_dir.joinpath('src/crclib')))
    import mycrclib


def get_jointable_cached(n):
    jointable_dir = Path(__file__).parent.joinpath('jointables')
    jointable_dir.mkdir(exist_ok=True)
    fname = jointable_dir.joinpath(f"jointable_{n:08d}.np")
    if fname.is_file() and False:
        jointable = np.load(fname)
    else:
        if n % 2 == 0 and n > 4:
            halftable = get_jointable_cached(n//2)
            jointable = np.zeros_like(halftable)
            for idx in np.ndindex(halftable.shape):
                jointable[idx] = pycrc.join_crc_from_lookup(
                    halftable[idx], 0, halftable)
        else:
            jointable = pycrc.get_jointable(n)
        # jointable = pycrc.get_jointable(n)
        np.save(fname, jointable)
    return jointable


n = 1024*1024*1024
# n = n - n % 32


# np.random.seed(123)
message = np.random.randint(0, 256, (n,), np.uint8)


args_norm = [message]

parallel_table = get_jointable_cached(n//16)
args_par = [message, parallel_table]

d_table = cp.asarray(pycrc.get_table_0())
d_message = cp.asarray(message.view(np.uint32))
size = d_message.size
d_result = cp.zeros((10000), np.uint32)
d_jointables = cp.asarray(np.stack(
    [get_jointable_cached(int((n//(16*256))*2**i))
     for i in range(8)]
))

args_cuda = [d_message.data.ptr, size,
             d_table.data.ptr, d_jointables.data.ptr,
             d_result.data.ptr, parallel_table]

functions = [
    # [pycrc.crc32,                       norm_args],
    # [pycrc.crc32_lookup,                norm_args],
    # [pycrc.crc32_jit,                   args_norm],
    [pycrc.crc32_lookup_jit,            args_norm],
    # [mycrclib.get_crc,                  args_norm],
    [mycrclib.get_crc_lookup,           args_norm],

    [pycrc.crc32_parallel,              args_par],
    [mycrclib.get_crc_lookup_parallel,  args_par],

    [mycrclib.get_crc_cuda_fast,        args_cuda],
]

results = []
for (func, args) in functions:
    output, time = timeit(func, *args, times=1)
    name = str(func)
    print(f"{name: <80}: {output: <12}, {time}")
    results.append((name, output, time))
