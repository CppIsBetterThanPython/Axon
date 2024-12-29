#include <vector>
#include <tuple>

#include "NetworkBackProp.h"

using Parameters = Network::Parameters;

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
vector<vector<vector<double>>> NetworkBackProp::differentiate(vector<double> expectedAnswers) {
    // How the network would like the node data to change, used to calculate weight and bias gradients
    vector<vector<double>> nodeGradientVector(networkSize);
    // Gradient of the controlable values (weights, biases)
    vector<vector<vector<double>>> controlableGradients(networkSize);
    for (int i = 0; i < networkSize; i++) {
        controlableGradients[i] = vector<vector<double>>(structure[i]);
    }

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
    for (size_t i = networkSize - 1; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i]; j++) {
            //for the amount of weights in the current node
            for (size_t k = 0; k < structure[i - 1]; k++) {
                double weightGradient = getWeightGradient(nodeGradientVector[i][j], i, j, k);
                controlableGradients[i][j].push_back(weightGradient);
            }

            double biasGradient = getBiasGradient(nodeGradientVector[i][j], i, j);
            controlableGradients[i][j].push_back(biasGradient);
        }
    }

    return controlableGradients;
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

tuple< vector<vector<vector<double>>>, double, bool> NetworkBackProp::Test(const vector<vector<double>>& testExample) {
    input(testExample[0]);
    calculate();

    vector<vector<vector<double>>> gradient = differentiate(testExample[1]);
    double cost = getCost(testExample[1]);

    bool isCorrect = false;
    if (isAnswerCorrect(testExample[1])) {
        isCorrect = true;
    }

    return { gradient, cost, isCorrect };
}

tuple< vector<vector<vector<double>>>, double, double> NetworkBackProp::TestSet(const vector<vector<vector<double>>>& testSet) {
    double averageCorrect = 0.0;
    double averageCost = 0.0;
    vector<vector<vector<double>>> averageGradient(networkSize);

    for (size_t i = 1; i < networkSize; i++) {
        averageGradient[i] = vector<vector<double>>(structure[i]);
        for (size_t j = 0; j < structure[i]; j++) {
            averageGradient[i][j] = vector<double>(structure[i - 1] + 1);
        }
    }

    for (vector<vector<double>> test : testSet) {
        auto testResult = Test(test);

        vector<vector<vector<double>>> gradient = get<0>(testResult);
        double cost = get<1>(testResult);
        bool isCorrect = get<2>(testResult);

        if (isCorrect) {
            averageCorrect++;
        }

        averageCost += cost;

        //for the amount of layers with data
        for (size_t i = networkSize - 1; i > 0; i--) {
            //for the amount of nodes in the current layer
            for (size_t j = 0; j < structure[i]; j++) {
                //for the amount of weights in the current node
                for (size_t k = 0; k < structure[i - 1] + 1; k++) {
                    averageGradient[i][j][k] += gradient[i][j][k];
                }
            }
        }
    }

    // Average gradient sum
    for (size_t i = networkSize - 1; i > 0; i--) {
        // For the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i]; j++) {
            // For the amount of weights in the current node
            for (size_t k = 0; k < structure[i - 1] + 1; k++) {
                averageGradient[i][j][k] /= testSet.size();
            }
        }
    }
    averageCost /= testSet.size();
    averageCorrect /= testSet.size();


    return { averageGradient, averageCost, averageCorrect };
}

// Removes gradient from the weights and biases
void NetworkBackProp::alterByGradient(const vector<vector<vector<double>>>& averageGradient, double learningRate) {
    //for the amount of layers with data
    for (size_t i = networkSize - 1; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (size_t j = 0; j < structure[i]; j++) {
            //for the amount of weights in the current node
            for (size_t k = 0; k < structure[i - 1]; k++) {

                layers[i][j][k] -= averageGradient[i][j][k] * learningRate;
                //cout << i << " " << j << " " << k << endl;
                //cout << layers[i][j][k] << " " << gradientSum[i][j][k] * learningRate << endl;
            }
            //cout << i << " " << j << " bias" << endl;
            layers[i][j].Bias -= averageGradient[i][j][structure[i] - 1] * learningRate;
            //cout << layers[i][j].Bias << " " << gradientSum[i][j][structure[i]-1] * learningRate << endl;
        }
    }
}

// Returns cost then accuracy
vector<double> NetworkBackProp::improveNetworkBackPropogation(const vector<vector<vector<double>>>& testSet, double learningRate) {

    auto averageTestResult = TestSet(testSet);

    vector<vector<vector<double>>> averageGradient = get<0>(averageTestResult);
    double averageCost = get<1>(averageTestResult);
    double averageCorrect = get<2>(averageTestResult);

    alterByGradient(averageGradient, learningRate);

    return { averageCost, averageCorrect };
}