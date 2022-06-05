#pragma once
// Error Checking Function
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort) exit(code);
    }
}

__global__ void crc_cuda(unsigned *message, int num_dword, unsigned *crc_table,
                         unsigned *crc_out);

__global__ void crc_cuda_fast(unsigned *d_message, int num_dword,
                              unsigned *d_crc_table, unsigned *d_join_tables,
                              unsigned *d_crc_out);