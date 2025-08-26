
cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB_RECURSE SOURCES "src/*.cpp")
add_library(nnlib STATIC ${SOURCES})

target_include_directories(nnlib
PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src/detail
)

# TODO: Add tests and install targets if needed.

target_precompile_headers(nnlib PRIVATE src/detail/pch.h)
