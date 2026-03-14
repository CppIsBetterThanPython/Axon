#pragma once

#include "NetworkBackPropBase.hpp"
#include "NetworkGPU.hpp"

namespace axon {

    class NetworkBackPropGPU : public NetworkBackPropBase, public NetworkGPU {
    private:

        NetworkBackPropGPU(Parameters& parameters);

        std::vector<std::unique_ptr<cl::Buffer>> layerBuffers;

        void backPropCalculate(size_t batchSize);
        inline void calculateGradients(const cl::Buffer& expectedBuffer, double learningRate, size_t batchSize, size_t prevBatchSize);
    public:

        TestResult TestSet(const std::vector<Test>& testSet) override;
        TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;
    };

}