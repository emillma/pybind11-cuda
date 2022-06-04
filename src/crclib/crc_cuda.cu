#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

#define poly 0x04c11db7
#define threads 320

__device__ void add_dword(unsigned &prev_crc, unsigned new_dword,
                          unsigned *table_0) {
    unsigned char *bview = reinterpret_cast<unsigned char *>(&new_dword);
    for (int i = 0; i < 4; i++) {
        prev_crc = (0xffffff & (prev_crc)) << 8 ^
                   table_0[bview[i] ^ ((prev_crc) >> 24)];
    };
}

// Simple CUDA kernel
__global__ void crc_cuda(unsigned *message, int num_dword, unsigned *table,
                         unsigned *crc_out) {
    __shared__ unsigned s_table[256];

    __shared__ unsigned s_crc[threads];
    __shared__ unsigned s_data[threads * 32];

    unsigned char *s_data_asbytes = reinterpret_cast<unsigned char *>(s_data);

    unsigned thread = threadIdx.x;
    unsigned windex = thread % 32;
    unsigned warpstart = thread - windex;
    unsigned chunksize = num_dword / threads;

    s_crc[thread] = 0;
    if (thread < 256) {
        s_table[thread] = table[thread];
    }

    for (int i = 0; i < chunksize; i += 32) {
        for (int j = 0; j < 32; j++) {
            s_data[(warpstart + j) * 32 + windex] =
                message[(warpstart + j) * chunksize + i + windex];
        }
        __syncthreads();

        for (int j = 0; j < 32 * 4; j++) {
            s_crc[thread] = s_crc[thread] << 8 ^
                            s_table[s_data_asbytes[thread * 32 * 4 + j] ^
                                    s_crc[thread] >> 24];
        }

        __syncthreads();
    }
    crc_out[thread] = s_crc[thread];
}
