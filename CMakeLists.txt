cmake_minimum_required(VERSION 3.2)

project(opencl_practice)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (UNIX)
    set(CMAKE_PREFIX_PATH /opt/amdgpu-pro)
    set(CMAKE_LIBRARY_ARCHITECTURE x86_64-linux-gnu)
endif()
find_package(OpenCL 2.0 REQUIRED)
find_package(HDF5 1.10 REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

project(hll_riemann)
if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -g")
endif()
add_executable(hll_riemann hll_riemann/main.cpp)
target_link_libraries(hll_riemann ${OpenCL_LIBRARIES} ${HDF5_LIBRARIES})