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
unsigned py_get_crc_cuda_fast(long pycuvec, int len, long crc_table,
                              long cuda_join_tables, long result,
                              py::array_t<unsigned> final_table) {
    unsigned *d_vec = reinterpret_cast<unsigned *>(pycuvec);
    unsigned *d_crc_table = reinterpret_cast<unsigned *>(crc_table);
    unsigned *d_join_tables = reinterpret_cast<unsigned *>(cuda_join_tables);
    unsigned *d_res = reinterpret_cast<unsigned *>(result);

    py::buffer_info final_table_buf = final_table.request();
    unsigned *final_table_ptr = static_cast<unsigned *>(final_table_buf.ptr);

    unsigned res[16];
    // Run kernel on 1M elements on the GPU
    int numBlocks = 16;
    int blockSize = 256;
    crc_cuda_fast<<<numBlocks, blockSize>>>(d_vec, len, d_crc_table,
                                            d_join_tables, d_res);

    cudaMemcpy(res, d_res, 16 * sizeof(unsigned), cudaMemcpyDeviceToHost);
    unsigned crc = 0;
    for (int i = 0; i < 16; i++) {
        crc = join_crc_from_lookup(crc, res[i], final_table_ptr);
    }
    // Wait for GPU to finish before accessing on host
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return crc;
}

PYBIND11_MODULE(mycrclib, m) {
    m.def("get_crc", &py_get_crc);
    m.def("get_crc_lookup", &py_get_crc_lookup);
    m.def("get_crc_lookup_parallel", &py_get_crc_lookup_parallel);
    m.def("get_crc_cuda_fast", &py_get_crc_cuda_fast);
}
