#pragma once

#include "NeuralNet.h"

// Back propogation involves partial derivative, including that of the sigmoid function
static inline double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

class NetworkBackProp :
    public Network
{
public:
    using TestData = std::vector<double>;
    using Answer = std::vector<double>;
    using Test = std::pair<TestData, Answer>;

    double getCost(std::vector<double> expectedAnswers);

    Parameters differentiate(std::vector<double> expectedAnswers);

private:
    double getNodeGradient(const std::vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos);
    double getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos);

public:
    std::tuple< Parameters, double, bool>   RunTrain(const std::pair<TestData, Answer>& testExample);
    std::tuple< Parameters, double, double> TrainSet(const std::vector<Test>& testSet);
    std::tuple< double, bool>   RunTest(const std::pair<TestData, Answer>& testExample);
    std::tuple< double, double> TestSet(const std::vector<Test>& testSet);

private:
    //removes average gradient from the weights and biases
    void alterByGradient(const Parameters& averageGradient, double learningRate);

public:
    //returns cost then accuracy
    std::pair<double, double> trainNetworkBackPropogation(const std::vector<Test>& testSet, double learningRate);
    //returns cost then accuracy
    std::pair<double, double> testNetworkBackPropogation(const std::vector<Test>& testSet);
};