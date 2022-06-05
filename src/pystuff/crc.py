import numpy as np
import time
import numba as nb
from pathlib import Path
from conf import STIM_DG_LEN, STIM_DG_PER_MES, CRC_IDX, DUMMY_BYTES
from typing import Callable

from collections.abc import Iterable
import pickle
u8 = np.uint8
u32 = np.uint32


def jit_from(func):
    def inner(ignored_func):
        return nb.njit(func)
    return inner


def crc32(message: Iterable[int],
          seed: int = 0):
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


@jit_from(crc32)
def crc32_jit(message: 'np.ndarray[:]', seed: int):
    ...


def get_table_0():
    table = np.empty(256, np.uint32)
    for i in range(256):
        bflip = crc32_jit(np.array([i], np.uint8), 0)
        table[i] = bflip
    return table


TABLE_ARR = get_table_0()


def crc32_lookup(message, crc=0):
    for val in message:
        crc = (0xffffff & crc) << 8 ^ TABLE_ARR[val ^ (crc >> 24)]
    return crc


@jit_from(crc32_lookup)
def crc32_lookup_jit(message: 'np.ndarray[:]', seed: int):
    ...


@nb.njit
def join_crc_from_lookup(crc1, crc2, join_table):
    crcout = crc2
    for i in range(4):
        crcout ^= join_table[i, crc1 >> i*8 & 0xff]
    return crcout


@nb.njit(parallel=True)
def crc32_parallel(buf, join_table, crc=u32(0)):
    splits = 16
    tmp = np.empty(splits, np.uint32)
    size = buf.shape[0]
    step = size/splits
    for i in nb.prange(splits):
        tmp[i] = crc32_lookup_jit(buf[step*i:step*(i+1)])
    for i in range(splits):
        crc = join_crc_from_lookup(crc, tmp[i], join_table)
    return crc


@nb.njit(parallel=True)
def get_jointable(dist):
    table = np.empty((4, 256), np.uint32)
    for i in nb.prange(4):
        for j in nb.prange(256):
            table[i, j] = crc32_lookup_jit(
                np.zeros(dist, np.uint8), j << i*8)
    return table


def extend_flip(bit, dist):
    crc = 1 << bit
    flip = crc32_lookup_jit(np.zeros(dist, np.uint8), crc)
    return flip


def join(crc1, crc2, dist):
    crcout = crc2
    for i in range(32):
        if crc1 >> i & 1:
            crcout = crcout ^ extend_flip(i, dist)
    return crcout
