add_library(mycrclib SHARED
  crcfuncs.cpp)

#target_include_directories(gpu_library PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(mycrclib PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(mycrclib PUBLIC
  Python3::Python)

# set what comes before the library name, here no "lib"
set_target_properties(mycrclib PROPERTIES PREFIX "")
