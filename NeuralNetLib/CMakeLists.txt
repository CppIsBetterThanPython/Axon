
cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OPENCL_ROOT "D:/Program Files/vcpkg/packages/opencl_x64-windows")

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_library(nnlib STATIC ${SOURCES})

target_include_directories(nnlib
PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/detail>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(nnlib PRIVATE "${OPENCL_ROOT}/include")

# Install targets

target_link_libraries(nnlib PRIVATE "${OPENCL_ROOT}/lib/OpenCL.lib")

install(TARGETS nnlib
        EXPORT nnlibTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        CONFIGURATIONS Debug Release
)

# TODO: Add tests and install targets if needed.

target_precompile_headers(nnlib PRIVATE src/detail/pch.h)
