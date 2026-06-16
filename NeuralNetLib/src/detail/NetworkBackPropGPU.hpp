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

	void saveBuffers() override;
        void loadBuffers();

        TestResult TestSet(const std::vector<Test>& testSet) override;
        TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;

	inline size_t maxWeightSize() {
		size_t maxWeightSize = 0;

		for (size_t i = 0; i < size(); i++)
			maxWeightSize = std::max(maxWeightSize, getStructure(i) * getStructure(i+1));

		return maxWeightSize;
	}
    };

}
