#include <tuple>

#include "pch.h"
#include "NetworkBackProp.h"

double NetworkBackProp::getCost(vector<double> expectedAnswers) {
    if (expectedAnswers.size() != (*outputLayer).size) {
        throw std::out_of_range("More answers than expected");
    }

    double cost = 0;
    vector<double> actualAnswers = getAnswerVector();

    for (int i = 0; i < (*outputLayer).size; i++) {
        cost += pow((actualAnswers[i] - expectedAnswers[i]), 2);
    }

    return cost;
}

// Returns a vector of how each bias and weight would like to be changed
Parameters NetworkBackProp::differentiate(vector<double> expectedAnswers) {
    // How the network would like the node data to change, used to calculate weight and bias gradients
    vector<vector<double>> nodeGradientVector(networkSize);
    // Gradient of the controlable values (weights, biases)
    Parameters parameterGradients = this->EmptyParameters();

    //calculates input layer gradients
    for (int i = 0; i < structure[networkSize - 1]; i++) {
        double nodeGradient = 2 * (layers[networkSize - 1][i].data - expectedAnswers[i]);
        nodeGradientVector[networkSize - 1].push_back(nodeGradient);
    }

    //for the amount of hidden layers
    for (size_t i = networkSize - 2; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i]; j++) {
            double nodeGradient = getNodeGradient(nodeGradientVector[i + 1], j, i);
            nodeGradientVector[i].push_back(nodeGradient);
        }
    }

    //for the amount of layers with data
    for (size_t i = 1; i < networkSize; i++) {

        //for the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i]; j++) {

            //for the amount of weights in the current node
            for (size_t k = 0; k < structure[i - 1]; k++)
                parameterGradients[i - 1][j][k] = getWeightGradient(nodeGradientVector[i][j], i, j, k);
 
            parameterGradients[i - 1][j][structure[i - 1]] = getBiasGradient(nodeGradientVector[i][j], i, j);
        }
    }

    return parameterGradients;
}

//input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double NetworkBackProp::getNodeGradient(const vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos) {
    double nodeGradient = 0.0;

    //  nL-1
    //  \    dC0   daj
    //  /    --- x ---
    //  j=0  daj   dak
    for (int j = 0; j < nextLayerNodeGradients.size(); j++) {

        //z
        double preSigmoidData = layers[currentLayerPos + 1][j].preSigmoidData;

        //wjk
        double currentLayerNodeWeight = layers[currentLayerPos + 1][j][nodePos];

        // daj
        // ---
        // dak
        double derivativeAjToAk = SigmoidDerivative(preSigmoidData) * currentLayerNodeWeight;

        // dC0   daj
        // --- x ---
        // daj   dak
        nodeGradient += nextLayerNodeGradients[j] * derivativeAjToAk;
    }

    return nodeGradient;
}

//input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double NetworkBackProp::getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos) {

    double preSigmoidNodeData = layers[layerPos][nodePos].preSigmoidData;

    //The data that the weight is multiplied by is part of the derivative
    double currentLayerNodeData = layers[layerPos - 1][weightPos].data;

    // daj
    // ---
    // dwjk
    double derivativeAjToWjk = SigmoidDerivative(preSigmoidNodeData) * currentLayerNodeData;

    // dC0   daj
    // --- x ---
    // daj   dwjk
    double gradient = nextLayerNodeGradient * derivativeAjToWjk;

    return gradient;
}

double NetworkBackProp::getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos) {

    double preSigmoidNodeData = layers[layerPos][nodePos].preSigmoidData;

    // daj
    // ---
    // dbj
    double derivativeAjToBj = SigmoidDerivative(preSigmoidNodeData);

    // dC0   daj
    // --- x ---
    // daj   dbj
    double gradient = nextLayerNodeGradient * derivativeAjToBj;

    return gradient;
}

tuple< Parameters, double, bool> NetworkBackProp::RunTrain(const std::pair<TestData, Answer>& testExample) {
    auto& [test, answer] = testExample;
    input(test);
    calculate();

    Parameters gradient = differentiate(answer);
    double cost = getCost(answer);

    bool isCorrect = false;
    if (isAnswerCorrect(answer)) {
        isCorrect = true;
    }

    return { gradient, cost, isCorrect };
}

tuple< Parameters, double, double> NetworkBackProp::TrainSet(const vector<Test>& testSet) {
    double averageCorrect = 0.0;
    double averageCost = 0.0;
    Parameters averageGradient = this->EmptyParameters();

    for (Test test : testSet) {
        auto testResult = RunTrain(test);

        Parameters gradient = get<0>(testResult);
        double cost = get<1>(testResult);
        bool isCorrect = get<2>(testResult);

        if (isCorrect) {
            averageCorrect++;
        }

        averageCost += cost;

        //for the amount of layers with data
        for (size_t i = 0; i < networkSize - 1; i++) {

            //for the amount of nodes in the current layer
            for (size_t j = 0; j < structure[i + 1]; j++) {

                //for the amount of weights in the current node
                for (size_t k = 0; k < structure[i]; k++)
                    averageGradient[i][j][k] += gradient[i][j][k];
            }
        }
    }

    // Average gradient sum
    for (size_t i = 0; i < networkSize - 1; i++) {
        // For the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i + 1]; j++) {
            // For the amount of weights in the current node
            for (size_t k = 0; k < structure[i]; k++) {
                averageGradient[i][j][k] /= testSet.size();
            }
        }
    }
    averageCost /= testSet.size();
    averageCorrect /= testSet.size();


    return { averageGradient, averageCost, averageCorrect };
}

tuple< double, bool> NetworkBackProp::RunTest(const std::pair<TestData, Answer>& testExample) {
    auto& [test, answer] = testExample;
    input(test);
    calculate();

    double cost = getCost(answer);

    bool isCorrect = false;
    if (isAnswerCorrect(answer)) {
        isCorrect = true;
    }

    return { cost, isCorrect };
}

tuple< double, double> NetworkBackProp::TestSet(const vector<Test>& testSet) {
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


    return { averageCost, averageCorrect };
}

// Removes gradient from the weights and biases
void NetworkBackProp::alterByGradient(const Parameters& averageGradient, double learningRate) {
    //for the amount of layers with data
    for (size_t i = 0; i < networkSize - 1; i++) {
        //for the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i + 1]; j++) {
            //for the amount of weights in the current node
            for (size_t k = 0; k < structure[i]; k++) {

                layers[i + 1][j][k] -= averageGradient[i][j][k] * learningRate;
            }
            layers[i + 1][j].Bias -= averageGradient[i][j][structure[i] - 1] * learningRate;
        }
    }
}

// Returns cost then accuracy
std::pair<double, double> NetworkBackProp::trainNetworkBackPropogation(const vector<Test>& testSet, double learningRate) {

    auto averageTestResult = TrainSet(testSet);

    Parameters averageGradient = get<0>(averageTestResult);
    double averageCost = get<1>(averageTestResult);
    double averageCorrect = get<2>(averageTestResult);

    alterByGradient(averageGradient, learningRate);

    return { averageCost, averageCorrect };
}


// Returns cost then accuracy. Does not alter network.
std::pair<double, double> NetworkBackProp::testNetworkBackPropogation(const vector<Test>& testSet) {

    auto averageTestResult = TestSet(testSet);

    double averageCost = get<0>(averageTestResult);
    double averageCorrect = get<1>(averageTestResult);

    return { averageCost, averageCorrect };
}