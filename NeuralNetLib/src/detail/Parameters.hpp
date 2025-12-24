#pragma once

#include "pch.h"

namespace axon {

    // TODO: Possibly store seed
    // TODO: Dynamically split the 1d vector
    // TODO: Store the activation and output function
    // 
    // Structure to store basic parameters of the network
    struct Parameters {
    private:
        std::vector<double> weightsData;
        std::vector<double> biasesData;

        bool isInitialised;
    public:
        std::vector<std::vector<std::span<double>>> weights;
        std::vector<std::span<double>> biases;
        size_t size;
        std::vector<size_t> structure;

        friend bool saveParameters(const Parameters& parameters, const std::filesystem::path& filePath);

        friend Parameters getParameters(const std::filesystem::path& filePath);

        Parameters(std::vector<size_t> structure);
        Parameters(Parameters&& other) noexcept;
        Parameters(const Parameters& other);

        Parameters& operator=(const Parameters& other);

        template<typename T>
        void initParameters(T& randomEngine) {
            for (size_t layerID = 0; layerID < size; layerID++)
                for (std::span<double>& node : weights[layerID])
                    for (double& weight : node)
                        weight = XavierInitialization(structure[layerID], structure[layerID + 1], randomEngine);

            for (double& bias : biasesData)
                bias = 0.1;

            isInitialised = true;
        }

        bool getIsInitialised() const { return isInitialised; }

        void moveSpans();

        Parameters operator+(Parameters other) const;
        Parameters operator-(Parameters other) const;

        template<typename T>
        Parameters operator*(T scalar) const {
            Parameters scaled = Parameters(structure);

            for (int i = 0; i < weightsData.size(); i++) {
                scaled.weightsData[i] = this->weightsData[i] * scalar;
            }

            for (int i = 0; i < biasesData.size(); i++) {
                scaled.biasesData[i] = this->biasesData[i] * scalar;
            }

            return scaled;
        }

        template<typename T>
        Parameters operator/(T scalar) const {
            Parameters scaled = Parameters(structure);

            for (int i = 0; i < weightsData.size(); i++) {
                scaled.weightsData[i] = this->weightsData[i] / scalar;
            }

            for (int i = 0; i < biasesData.size(); i++) {
                scaled.biasesData[i] = this->biasesData[i] / scalar;
            }

            return scaled;
        }
        bool operator==(const Parameters& other) const;
    };

}