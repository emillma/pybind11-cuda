import numpy as np
import time
import numba as nb
from pathlib import Path
from conf import STIM_DG_LEN, STIM_DG_PER_MES, CRC_IDX, DUMMY_BYTES


from collections.abc import Iterable
import pickle
u8 = np.uint8
u32 = np.uint32


poly = 0x104c11db7
TABLE = []
for i in range(256):
    bflip = i << 24
    for _ in range(8):
        if bflip & 0x80000000:
            bflip = (bflip << 1) ^ poly
        else:
            bflip <<= 1
    TABLE.append(bflip)
TABLE_ARR = np.array(TABLE, np.uint32)
# TABLE = TABLE_ARR


def crc32mpeg2(message: Iterable[int],
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


@nb.njit
def crc32mpeg2_jited(message, seed=0xffffffff):
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


def crc32mpeg2_lookup(message, crc=0xffffffff):
    for val in message:
        crc = (0xffffff & crc) << 8 ^ TABLE[val ^ (crc >> 24)]
    return crc


@nb.njit
def crc32mpeg2_lookup_jited(buf, crc=u32(0xffffffff)):
    for val in buf:
        crc = ((u32(0xffffff) & crc) << u8(8)
               ^ TABLE_ARR[val ^ (crc >> u8(24))])
    return crc


def check_all(checker, data):
    out = []
    for dg in range(0, STIM_DG_PER_MES):
        step = dg*STIM_DG_LEN
        message_part = data[step:step+STIM_DG_LEN]
        crc = checker(message_part[:CRC_IDX] + DUMMY_BYTES)
        check = int.from_bytes(bytes(message_part[CRC_IDX:CRC_IDX+4]), 'big')
        out.append(crc == check)
    return out


def check_all_tupled(checker, data):
    out = np.empty(STIM_DG_PER_MES, dtype=np.bool_)
    # crc = np.empty(STIM_DG_PER_MES, dtype=np.uint32)
    for dg in nb.prange(0, STIM_DG_PER_MES):
        step = dg*STIM_DG_LEN
        message_part = data[step:step+STIM_DG_LEN]
        crc = checker(message_part[:CRC_IDX])
        crc = checker(np.array([0, ], np.uint32), crc)
        check = np.sum(message_part[CRC_IDX:CRC_IDX+4]
                       * 2**np.arange(24, -1, -8))
        out[dg] = (crc == check)
    return out


@nb.njit(cache=True, parallel=False)
def check_all_jited(data):
    out = np.empty(STIM_DG_PER_MES, dtype=np.bool_)
    # crc = np.empty(STIM_DG_PER_MES, dtype=np.uint32)
    for dg in nb.prange(0, STIM_DG_PER_MES):
        step = dg*STIM_DG_LEN
        message_part = data[step:step+STIM_DG_LEN]
        crc = crc32mpeg2_lookup_jited(message_part[:CRC_IDX])
        crc = crc32mpeg2_lookup_jited(np.array([0, ], np.uint32), crc)
        check = np.sum(message_part[CRC_IDX:CRC_IDX+4]
                       * 2**np.arange(24, -1, -8))
        out[dg] = (crc == check)
    return out


def timeit(func, *args, times=100):
    func(*args)
    t0 = time.perf_counter()
    for i in range(times):
        func(*args)
    t_final = time.perf_counter()
    return (t_final - t0)/times
