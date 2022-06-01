#include <sstream>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(gpu_library, m)
{
    m.def("multiply_with_scalar", map_array<double>);
}