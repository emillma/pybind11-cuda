#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

#define poly 0x04c11db7
#define threads 256

__device__ void include_data(unsigned &prev_crc, unsigned char *new_bytes,
                             unsigned *table, int num) {
    for (int j = 0; j < num; j++) {
        prev_crc = prev_crc << 8 ^ table[new_bytes[j] ^ prev_crc >> 24];
    }
}

// Simple CUDA kernel
__global__ void crc_cuda1(unsigned *message, int num_dword, unsigned *table,
                          unsigned *crc_out) {
    auto byte_view = reinterpret_cast<unsigned char *>(message);
    unsigned thread = threadIdx.x;
    unsigned chunksize = 4 * num_dword / blockDim.x;

    unsigned crc = 0;
    include_data(crc, &byte_view[thread * chunksize], table, chunksize);
    crc_out[thread] = crc;
}
__global__ void crc_cuda(unsigned *message, int num_dword, unsigned *table,
                         unsigned *crc_out) {
    __shared__ unsigned s_table[256];
    __shared__ unsigned s_crc[1024];

    auto byte_view = reinterpret_cast<unsigned char *>(message);
    unsigned thread = threadIdx.x;
    unsigned chunksize = 4 * num_dword / blockDim.x;

    s_crc[thread] = 0;
    if (thread < 256) {
        s_table[thread] = table[thread];
    }

    include_data(s_crc[thread], &byte_view[thread * chunksize], s_table,
                 chunksize);
    crc_out[thread] = s_crc[thread];
}

__global__ void crc_cuda_opt1(unsigned *message, int num_dword, unsigned *table,
                              unsigned *crc_out) {
    __shared__ unsigned s_table[256];
    __shared__ unsigned s_crc[threads];
    __shared__ unsigned s_data[threads * 32];

    auto *s_data_asbytes = reinterpret_cast<unsigned char *>(s_data);
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
        include_data(s_crc[thread], &s_data_asbytes[thread * 32 * 4], s_table,
                     32 * 4);
    }
    crc_out[thread] = s_crc[thread];
}
