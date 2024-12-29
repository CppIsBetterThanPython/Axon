#pragma once

#include <vector>

#include "NeuralNet.h"

// Back propogation involves partial derivative, including that of the sigmoid function
static inline double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

class NetworkBackProp :
    public Network
{
public:
    using Network::Network;

    double getCost(vector<double> expectedAnswers);

    vector<vector<vector<double>>> differentiate(vector<double> expectedAnswers);

private:
    double getNodeGradient(const vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos);
    double getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos);

public:
    tuple< vector<vector<vector<double>>>, double, bool>   Test(const vector<vector<double>>& testExample);
    tuple< vector<vector<vector<double>>>, double, double> TestSet(const vector<vector<vector<double>>>& testSet);

private:
    //removes average gradient from the weights and biases
    void alterByGradient(const vector<vector<vector<double>>>& averageGradient, double learningRate);

public:
    //returns cost then accuracy
    vector<double> improveNetworkBackPropogation(const vector<vector<vector<double>>>& testSet, double learningRate);
};

