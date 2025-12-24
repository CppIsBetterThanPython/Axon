#pragma once

#include "pch.h"

#include "Parameters.hpp"
#include "NetworkBase.hpp"

namespace cl {
    class Buffer;
}

class GPU;

namespace axon {

    // Interface for networks with GPU, never exists on its own
    class NetworkGPU : public NetworkBase {
    protected:
        std::vector<std::unique_ptr<cl::Buffer>> WeightBuffers;
        std::vector<std::unique_ptr<cl::Buffer>> BiasBuffers;

        size_t batchSize = 1;

        std::unique_ptr<cl::Buffer> inputBuffer;
        std::unique_ptr<cl::Buffer> outputBuffer;

        std::unique_ptr<GPU> gpu;

        Parameters& parameters;
    public:

        NetworkGPU(Parameters& parameters);

        ~NetworkGPU();

        void input(const std::vector<std::vector<double>>&) override;
        void input(const std::vector<double>&) override;
        void calculate() override;
        [[nodiscard]] std::vector<double> getAnswerVector() const override;
        [[nodiscard]] std::vector<std::vector<double>> getAnswerVectors() const override;

        void loadBuffers();
        void saveBuffers();

        inline size_t largestLayer() const {
            size_t curLargestLayer = 0;

            for (size_t i = 0; i < size(); i++)
                curLargestLayer = std::max(curLargestLayer, getStructure(i));

            return curLargestLayer;
        }

        inline size_t largestPassLayer() const {
            size_t curLargestLayer = 0;

            for (size_t i = 1; i < size(); i++)
                curLargestLayer = std::max(curLargestLayer, getStructure(i));

            return curLargestLayer;
        }
    };

}