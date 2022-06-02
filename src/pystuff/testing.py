import numpy as np
import sys
from pathlib import Path
import cupy as cp
import numba as ba
from numba import cuda
if True:
    build_dir = Path(__file__).parents[2].joinpath('build')
    sys.path.append(str(build_dir.joinpath('src/crclib')))
    import mycrclib

x = cp.ones(1, np.float32)
y = cp.ones(1, np.float32)

nbarray = cuda.to_device(np.ones(3, dtype=np.float32))


a = mycrclib.pyadd(len(x), x.data.ptr, y.data.ptr)
a = mycrclib.pyadd(len(x), x.data.ptr,
                   nbarray.__cuda_array_interface__['data'][0])
print(type(a))


def crc32mpeg2(message,
               seed: int = 0xffffffff):
    """
    Thanks to https://stackoverflow.com/questions/69332500/how-can-calculate-mpeg2-crc32-in-python
    """
    poly = 0x104c11db7
    crc = seed
    for val in message:
        crc ^= val << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
    return crc


a = np.random.randint(0, 255, (1000,), dtype=np.uint8)
a = np.zeros((4), dtype=np.uint8)
a = np.array([0, 0, 0, 0, 1], dtype=np.uint8)
print(mycrclib.crc1(a))
print(crc32mpeg2(a))
