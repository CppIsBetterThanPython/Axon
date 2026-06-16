cmake_minimum_required(VERSION 3.16)
project(Sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sandbox
	main.cpp
)

find_package(OpenCL REQUIRED)
target_link_libraries(sandbox OpenCL::OpenCL)
