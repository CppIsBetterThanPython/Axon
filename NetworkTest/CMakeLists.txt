# CMakeList.txt : CMake project for NetworkTest, include source and define
# project specific logic here.
#

# Add source to this project's executable.
cmake_minimum_required(VERSION 3.16)
project(NeuralNetTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(nntest ${SOURCES})

target_include_directories(nntest
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/detail
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(nntest PRIVATE nnlib)
# TODO: Add tests and install targets if needed.
