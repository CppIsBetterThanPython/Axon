// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.
#pragma once

// Standard containers
#include <tuple>
#include <vector>
#include <array>
#include <span>

#include <filesystem> // For file operations

#include <iostream> // For error logging

#include <memory> // For smart pointers
#include <algorithm>
#include <numeric> // For std::accumulate and other algorithms
#include <random> // For deterministic and high quality PRNG
#include <system_error> // For standard error codes

// My own utilities header with functions that would not look right in any single header file
// I will not change it often so it doesn't override the benifits of a pch
#include "utils.hpp"