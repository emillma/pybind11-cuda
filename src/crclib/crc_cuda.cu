#include <cuda_runtime.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

// Error Checking Function
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Simple CUDA kernel
__global__ void cuadd(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (index == 0)
        y[0] = x[0] + y[0];
}
