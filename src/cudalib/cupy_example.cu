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
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// Simple wrapper function to be exposed to Python
int pyadd(int N, long px, long py) {
    float *x = reinterpret_cast<float *>(px);
    float *y = reinterpret_cast<float *>(py);
    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cuadd<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return 0;
}

PYBIND11_MODULE(demolib, m) {
    m.doc() = "pybind11 example plugin";  // optional module docstring
    m.def("pyadd", &pyadd, "A function which adds two numbers");
}