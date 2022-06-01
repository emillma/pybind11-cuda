import numpy as np
import sys
from pathlib import Path
if True:
    build_dir = Path(__file__).parents[2].joinpath('build')
    sys.path.append(str(build_dir.joinpath('src/crclib')))
    import mycrclib


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
print(mycrclib.crc1(a))
print(crc32mpeg2(a))
