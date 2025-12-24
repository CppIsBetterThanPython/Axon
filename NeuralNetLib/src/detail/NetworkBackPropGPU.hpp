#pragma once

#include "NetworkBackPropBase.hpp"
#include "NetworkGPU.hpp"

// TODO: Uncomment this
/*
class NetworkBackPropGPU : public NetworkBackPropBase, public NetworkGPU {
private:

    NetworkBackPropGPU(Parameters& parameters);

    std::vector<std::unique_ptr<cl::Buffer>> layerrBuffers;

    void backPropCalculate();
public:

    TestResult TestSet(const std::vector<Test>& testSet) override;
    TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;
};
*/