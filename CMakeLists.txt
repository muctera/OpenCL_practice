cmake_minimum_required(VERSION 3.2)

project(opencl_practice)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (UNIX)
    set(CMAKE_PREFIX_PATH /opt/amdgpu-pro)
    set(CMAKE_LIBRARY_ARCHITECTURE x86_64-linux-gnu)
endif()
find_package(OpenCL 1.2 REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

project(hello)
add_executable(hello hello/hello.c)
target_link_libraries(hello ${OpenCL_LIBRARIES})
