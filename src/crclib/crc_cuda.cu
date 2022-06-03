#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

#define poly 0x04c11db7

constexpr __device__ unsigned get_lookup(int i) {
    unsigned bflip = i << 24;
    for (int j = 0; j < 8; j++) {
        if (bflip & 0x80000000) {
            bflip = (bflip << 1) ^ poly;
        } else {
            bflip = bflip << 1;
        }
    }
    return bflip;
}
// Simple CUDA kernel
__global__ void crc_cuda(unsigned char *message, int len, unsigned *crc_out) {
    __shared__ unsigned workmem[1024];
    __shared__ unsigned table[256];
    int thread = threadIdx.x;

    if (thread < 256) {
        table[thread] = get_lookup(thread);
    }
    workmem[thread] = 0;

    int stride = 1 + ((len - 1) / blockDim.x);
    for (int i = 0; i < stride; i++) {
        int idx = thread * stride + i;
        if (idx < len) {
            unsigned char val = message[idx];
            unsigned crc = workmem[thread];
            workmem[thread] = (0xffffff & crc) << 8 ^ table[val ^ static_cast<unsigned char>(crc >> 24)];
        }
    }
    if (thread == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            *crc_out ^= workmem[i];
        }
    }
}
