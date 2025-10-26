
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

install(DIRECTORY include/ DESTINATION include)

install(EXPORT nnlibTargets
        FILE nnlibConfig.cmake
        NAMESPACE nnlib::
        DESTINATION lib/cmake/nnlib
)

target_precompile_headers(nnlib PRIVATE src/detail/pch.h)

# Add Tests

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(nnbenchmark "benchmark/benchmark.cpp")

target_link_libraries(nnbenchmark
PRIVATE
    nnlib
    benchmark::benchmark
)

add_executable(nntest "tests/test.cpp")

target_compile_options(nntest PRIVATE
    $<$<CONFIG:Debug>:/Zi /Od>
)

target_link_libraries(
    nntest
PRIVATE
    nnlib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(nntest)

add_executable(test_utils "tests/testUtils.cpp")

target_compile_options(test_utils PRIVATE
    $<$<CONFIG:Debug>:/Zi /Od>
)

target_link_libraries(
    test_utils
PRIVATE
    nnlib
    GTest::gtest_main
)

gtest_discover_tests(test_utils)