#include <iostream>
#include <sstream>

// #include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cutest.cu"

namespace py = pybind11;

template <typename T>
bool crc1(py::array_t<T> vec) {
    py::buffer_info ha = vec.request();
    if (ha.ndim != 1) {
        throw std::runtime_error("wrong dim");
    }
    int size = ha.shape[0];
    unsigned int crc = 0xffffffff;
    unsigned int poly = 0x04c11db7;

    auto r = vec.unchecked();
    for (py::ssize_t i = 0; i < size - 4; i++) {
        unsigned int val = static_cast<unsigned int>(r(i)) << 24;
        crc = crc ^ val;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ poly;
            } else {
                crc = crc << 1;
            }
        }
    }
    unsigned int check = __builtin_bswap32(*reinterpret_cast<unsigned int *>(&static_cast<T *>(ha.ptr)[1]));
    return (check == crc);
}

PYBIND11_MODULE(mycrclib, m) {
    m.def("crc1", crc1<unsigned char>);
    m.def("pyadd", &pyadd, "A function which adds two numbers");
}
