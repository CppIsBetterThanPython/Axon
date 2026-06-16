#pragma once

#include "pch.h"

inline constexpr size_t defaultSeed = 5489u;

// TODO: Try to make constexpr
// To turn values into 0-1 range
static inline double Sigmoid(const double x) { return 1 / (1 + exp(-x)); }

// Following function for dynamic GPU kernel compilation
template<typename T>
constexpr const std::string typeName() { return "unknown"; }

template<>
constexpr const std::string typeName<float>() { return "f32"; }

template<>
constexpr const std::string typeName<double>() { return "f64"; }

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

// For random number to use in Xavier Initialization
template<typename T>
inline double RandomReal(T& randomEngine) {
    static std::uniform_real_distribution<double> distribution(-1, 1);

    return distribution(randomEngine);
}

// More efficient and stable initialisation
template<typename T>
static inline double XavierInitialization(size_t in, size_t out, T& randomEngine) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal(randomEngine) * range;  // Scaled random value
    return x;
}