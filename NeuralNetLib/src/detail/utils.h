#pragma once

#include "pch.h"

// TODO: Try to make constexpr
// To turn values into 0-1 range
static inline double Sigmoid(const double x) { return 1 / (1 + exp(-x)); }

template<typename C>
inline std::vector<typename C::value_type> flattenVector(const std::vector<C>& vec) {
    using T = typename C::value_type;
    std::vector<T> flattened;

    size_t size = std::accumulate(vec.begin(), vec.end(), size_t{ 0 }, [](size_t init, const C& container) -> size_t { return init += container.size(); });
    flattened.reserve(size);

    for (const C& memb : vec) flattened.insert(flattened.end(), memb.begin(), memb.end());

    return flattened;
}

// Gets the index of the largest element in the vector
template<typename T>
size_t getLargestID(const std::vector<T>& vec) {
    size_t currentLargestID = 0;

    for (size_t i = 0; i < vec.size(); i++)
        if (vec[i] > vec[currentLargestID])
            currentLargestID = i;

    return currentLargestID;
}

// TODO: use std::random 
// For random number to use in Xavier Initialization
inline double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

// More efficient and stable initialisation
static inline double XavierInitialization(size_t in, size_t out) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal() * range;  // Scaled random value
    return x;
}