#pragma once

#include "NetworkBackPropBase.hpp"
#include "NetworkCPU.hpp"

class NetworkBackPropCPU : public NetworkBackPropBase, public NetworkCPU {
private:
    using NetworkCPU::NetworkCPU;

    double getCost(std::vector<double> expectedAnswers);

    double getNodeGradient(const std::vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos);
    double getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos);

    Parameters differentiate(std::vector<double> expectedAnswers);

    std::tuple< Parameters, double, bool> RunTrain(const Test& test);
    std::tuple< double, bool>             RunTest(const Test& test);
public:
    TestResult TestSet(const std::vector<Test>& testSet) override;
    TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;

private:
    //removes average gradient from the weights and biases
    void alterByGradient(const Parameters& averageGradient, double learningRate);
};