#include <iostream>
#include <sstream>

// #include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
unsigned int crc1(py::array_t<T> vec) {
    py::buffer_info ha = vec.request();
    if (ha.ndim != 1) {
        throw std::runtime_error("wrong dim");
    }
    int size = ha.shape[0];

    return 32;

    // unsigned int crc = 0x11111111;
    // unsigned int poly = 0x04c11db7;
    // for (int i = 0; i < size; i++) {
    //     crc = crc ^ static_cast<unsigned int>(vec[i]);
    //     for (int j = 0; j < 8; j++) {
    //         if (crc & 0x80000000) {
    //             crc = (crc << 1) ^ poly;
    //         } else {
    //             crc = crc << 1;
    //         }
    //     }
    // }
    // return crc;
    // auto output = py::array_t<T>(4);
}

PYBIND11_MODULE(gpu_library, m) {
    m.def("crc1", crc1<unsigned char>);
}
