#pragma once

#include <vector>

#include "NeuralNet.h"

// Back propogation involves partial derivative, including that of the sigmoid function
static inline double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

class NetworkBackProp :
    public Network
{
public:
    using TestData = vector<double>;
    using Answer = vector<double>;
    using Test = std::pair<TestData, Answer>;

    double getCost(vector<double> expectedAnswers);

    Parameters differentiate(vector<double> expectedAnswers);

private:
    double getNodeGradient(const vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos);
    double getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos);

public:
    tuple< Parameters, double, bool>   RunTrain(const std::pair<TestData, Answer>& testExample);
    tuple< Parameters, double, double> TrainSet(const vector<Test>& testSet);
    tuple< double, bool>   RunTest(const std::pair<TestData, Answer>& testExample);
    tuple< double, double> TestSet(const vector<Test>& testSet);

private:
    //removes average gradient from the weights and biases
    void alterByGradient(const Parameters& averageGradient, double learningRate);

public:
    //returns cost then accuracy
    std::pair<double, double> trainNetworkBackPropogation(const vector<Test>& testSet, double learningRate);
    //returns cost then accuracy
    std::pair<double, double> testNetworkBackPropogation(const vector<Test>& testSet);
};

