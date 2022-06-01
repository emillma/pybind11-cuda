add_library(crclib SHARED
  crcfuncs.cpp)

#target_include_directories(gpu_library PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(crclib PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(crclib PUBLIC
  Python3::Python
  cudart)

# set what comes before the library name, here no "lib"
set_target_properties(crclib PROPERTIES PREFIX "")
