#include "NetworkBackProp.h"

using std::vector, std::tuple;

double NetworkBackPropCPU::getCost(vector<double> expectedAnswers) {
    if (expectedAnswers.size() != getStructure().back()) {
        throw std::out_of_range("More answers than expected");
    }

    double cost = 0;
    vector<double> actualAnswers = getAnswerVector();

    for (int i = 0; i < getStructure().back(); i++) {
        cost += pow((actualAnswers[i] - expectedAnswers[i]), 2);
    }

    return cost;
}

// Input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double NetworkBackPropCPU::getNodeGradient(const vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos) {
    double nodeGradient = 0.0;

    //  nL-1
    //  \    dC0   daj
    //  /    --- x ---
    //  j=0  daj   dak
    for (int j = 0; j < nextLayerNodeGradients.size(); j++) {

        // sigma(z)
        double data = nodeData[currentLayerPos + 1][j];

        // wjk
        double currentLayerNodeWeight = parameters.weights[currentLayerPos][j][nodePos];

        // daj
        // ---
        // dak
        double derivativeAjToAk = data * (1 - data) * currentLayerNodeWeight;

        // dC0   daj
        // --- x ---
        // daj   dak
        nodeGradient += nextLayerNodeGradients[j] * derivativeAjToAk;
    }

    return nodeGradient;
}

// Input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double NetworkBackPropCPU::getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos) {

    // sigma(z)
    double data = nodeData[layerPos][nodePos];

    //The data that the weight is multiplied by is part of the derivative
    double currentLayerNodeData = nodeData[layerPos - 1][weightPos];

    // daj
    // ---
    // dwjk
    double derivativeAjToWjk = data * (1 - data) * currentLayerNodeData;

    // dC0   daj
    // --- x ---
    // daj   dwjk
    double gradient = nextLayerNodeGradient * derivativeAjToWjk;

    return gradient;
}

double NetworkBackPropCPU::getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos) {
    
    double data = nodeData[layerPos][nodePos];

    // daj
    // ---
    // dbj
    double derivativeAjToBj = data * (1 - data);

    // dC0   daj
    // --- x ---
    // daj   dbj
    double gradient = nextLayerNodeGradient * derivativeAjToBj;

    return gradient;
}

// Returns a vector of how each bias and weight would like to be changed
Parameters NetworkBackPropCPU::differentiate(vector<double> expectedAnswers) {
    // How the network would like the node data to change, used to calculate weight and bias gradients
    vector<vector<double>> nodeGradientVector(size());
    // Gradient of the controlable values (weights, biases)
    Parameters parameterGradients = Parameters(getStructure());

    // Calculates input layer gradients
    for (size_t i = 0; i < getStructure().back(); i++) {
        double nodeGradient = 2 * (nodeData.back()[i] - expectedAnswers[i]);
        nodeGradientVector.back().push_back(nodeGradient);
    }

    // For the amount of hidden layers
    for (size_t i = size() - 2; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (size_t j = 0; j < getStructure()[i]; j++) {
            double nodeGradient = getNodeGradient(nodeGradientVector[i + 1], j, i);
            nodeGradientVector[i].push_back(nodeGradient);
        }
    }

    //for the amount of layers with data
    for (size_t i = 1; i < size(); i++) {

        //for the amount of nodes in the current layer
        for (size_t j = 0; j < getStructure()[i]; j++) {

            //for the amount of weights in the current node
            for (size_t k = 0; k < getStructure()[i - 1]; k++)
                parameterGradients.weights[i - 1][j][k] = getWeightGradient(nodeGradientVector[i][j], i, j, k);

            parameterGradients.biases[i - 1][j] = getBiasGradient(nodeGradientVector[i][j], i, j);
        }
    }

    return parameterGradients;
}

tuple< Parameters, double, bool> NetworkBackPropCPU::RunTrain(const Test& test) {
    input(test.input.values);
    calculate();

    Parameters gradient = differentiate(test.expected.values);
    double cost = getCost(test.expected.values);

    bool isCorrect = isAnswerCorrect(test.expected.values);

    return { gradient, cost, isCorrect };
}

TestResult NetworkBackPropCPU::TrainSet(const vector<Test>& testSet, double learningRate) {
    double averageCorrect = 0.0;
    double averageCost = 0.0;
    Parameters averageGradient = Parameters(getStructure());

    for (Test test : testSet) {
        auto testResult = RunTrain(test);

        Parameters gradient = get<0>(testResult);
        double cost = get<1>(testResult);
        bool isCorrect = get<2>(testResult);

        if (isCorrect) {
            averageCorrect++;
        }

        averageCost += cost;

        averageGradient = averageGradient + gradient;
    }

    averageGradient = averageGradient / testSet.size();
    averageCost /= testSet.size();
    averageCorrect /= testSet.size();

    alterByGradient(averageGradient, learningRate);

    return TestResult{ averageCost, averageCorrect };
}

tuple< double, bool> NetworkBackPropCPU::RunTest(const Test& test) {
    input(test.input.values);
    calculate();

    double cost = getCost(test.expected.values);

    bool isCorrect = isAnswerCorrect(test.expected.values);

    return { cost, isCorrect };
}

TestResult NetworkBackPropCPU::TestSet(const vector<Test>& testSet) {
    double averageCorrect = 0.0;
    double averageCost = 0.0;

    for (Test test : testSet) {
        auto testResult = RunTest(test);

        double cost = get<0>(testResult);
        bool isCorrect = get<1>(testResult);

        if (isCorrect) {
            averageCorrect++;
        }

        averageCost += cost;
    }

    averageCost /= testSet.size();
    averageCorrect /= testSet.size();


    return TestResult{ averageCost, averageCorrect };
}

// Removes gradient from the weights and biases
void NetworkBackPropCPU::alterByGradient(const Parameters& averageGradient, double learningRate) {
    parameters = parameters - averageGradient * learningRate;
}