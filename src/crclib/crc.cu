#include <iostream>
#include <sstream>

// #include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "crc_cpp.hpp"
#include "crc_cuda.cuh"

namespace py = pybind11;

unsigned int py_get_crc(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc(ptr, len);
}

unsigned int py_get_crc_lookup(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc_lookup(ptr, len);
}

unsigned int py_get_crc_parallel(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc_parallel(ptr, len, 8);
}

unsigned int py_get_crc_lookup_parallel(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    return get_crc_lookup_parallel(ptr, len, 8);
}

PYBIND11_MODULE(mycrclib, m) {
    m.def("get_crc", &py_get_crc);
    m.def("get_crc_lookup", &py_get_crc_lookup);
    m.def("get_crc_parallel", &py_get_crc_parallel);
    m.def("get_crc_lookup_parallel", &py_get_crc_lookup_parallel);
}
