#include <cuda_runtime.h>

#include <iostream>
#include <sstream>

template <typename T>
__global__ void kernel(T *vec, T scalar, int num_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        vec[idx] = vec[idx] * scalar;
    }
}

template <typename T>
void run_kernel(T *vec, T scalar, int num_elements) {
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(ceil((T)num_elements / dimBlock.x));

    kernel<T><<<dimGrid, dimBlock>>>(vec, scalar, num_elements);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream strstr;
        strstr << "run_kernel launch failed" << std::endl;
        strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
        strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
        strstr << cudaGetErrorString(error);
        throw strstr.str();
    }
}