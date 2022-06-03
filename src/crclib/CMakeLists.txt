add_library(mycrclib SHARED
  crc.cu
  crc_cpp.cpp
  crc_cuda.cu)

find_package(OpenMP)
target_include_directories(mycrclib PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(mycrclib PUBLIC
  Python3::Python
  cudart
  OpenMP::OpenMP_CXX)

# set what comes before the library name, here no "lib"
set_target_properties(mycrclib PROPERTIES PREFIX "")
