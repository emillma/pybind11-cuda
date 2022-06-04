#include <iostream>
#include <sstream>

// #include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "crc_cpp.hpp"
#include "crc_cuda.cuh"

namespace py = pybind11;

unsigned py_get_crc(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc(ptr, len);
}

unsigned py_get_crc_lookup(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc_lookup(ptr, len);
}

unsigned py_get_crc_lookup_parallel(py::array_t<unsigned char> vec,
                                    py::array_t<unsigned> table) {
    py::buffer_info vec_buf = vec.request();
    py::buffer_info table_buf = table.request();
    int len = vec_buf.shape[0];
    unsigned char *vec_ptr = static_cast<unsigned char *>(vec_buf.ptr);
    unsigned *table_ptr = static_cast<unsigned *>(table_buf.ptr);
    return get_crc_lookup_parallel(vec_ptr, len, table_ptr);
}

// Simple wrapper function to be exposed to Python
unsigned py_get_crc_cuda(long pycuvec, int len, long py_table, long result) {
    unsigned *d_vec = reinterpret_cast<unsigned *>(pycuvec);
    unsigned *d_table = reinterpret_cast<unsigned *>(py_table);
    unsigned *d_res = reinterpret_cast<unsigned *>(result);
    // Run kernel on 1M elements on the GPU
    int numBlocks = 1;
    int blockSize = 1024;

    crc_cuda<<<numBlocks, blockSize>>>(d_vec, len, d_table, d_res);
    // Wait for GPU to finish before accessing on host
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return 0;
}

PYBIND11_MODULE(mycrclib, m) {
    m.def("get_crc", &py_get_crc);
    m.def("get_crc_lookup", &py_get_crc_lookup);
    m.def("get_crc_lookup_parallel", &py_get_crc_lookup_parallel);
    m.def("get_crc_cuda", &py_get_crc_cuda);
}
