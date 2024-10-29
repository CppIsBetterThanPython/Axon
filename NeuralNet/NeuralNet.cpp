#include <cmath> //for basic math functions, for exponent used in sigmoid
#include <cstdlib> //for random numbers
#include <ctime> //so is this
#include <iostream>
#include <vector> // (._.) < Really? ... )
#include <deque>
#include <utility>
#include <tuple>//to return multiple different type values

#include "NeuralNet.h"

using namespace std;

//to turn values into 0-1 range
double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

//to backpropogate
double SigmoidDerivative(double x) {
    return Sigmoid(x)*(1-Sigmoid(x));
}

bool isInRange(double num, double lower, double upper) {
    bool x = (num >= lower && num <= upper);
    return (num >= lower && num <= upper);
}

//to set preliminary guesses for weights and biases
double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

//class Network
////////////////////////////////////////////////////////////////////////////////////////////////

//constructor
Network::Network(vector<int> Structure) {
    networkSize = Structure.size();
    structure = Structure;
    //defines amount of layers
    layers = new Layer[Structure.size()];

    for(int i = 0; i < Structure.size(); i++) {
        layers[i] = Layer(Structure[i], (i > 0) ? Structure[i-1] : 0);
    }

    inputLayer = &layers[0];
    outputLayer = &layers[networkSize - 1];
}

//input vector to set input node layer
void Network::input(vector<double> input) {
    if (input.size() == (*inputLayer).size) {
        for(int i = 0; i < (*inputLayer).size; i++) {
            (*inputLayer)[i].data = input[i];
        }
    }
}

//passes data through layers
void Network::calculate() {
    double preSigmoidTotal =    0;
    double prevLayerData =      0.0;
    double currentLayerWeight = 0.0;
    for (int currentLayerIndex = 1; currentLayerIndex < networkSize; currentLayerIndex++) {

        for (int nodeIndex = 0; nodeIndex < layers[currentLayerIndex].size; nodeIndex++) {

            for (int weightIndex = 0; weightIndex < layers[currentLayerIndex-1].size; weightIndex++) {
                //multiply previous nodes data by it's corresponding weight in the current node

                prevLayerData = layers[currentLayerIndex - 1].nodes[weightIndex].data;
                currentLayerWeight = layers[currentLayerIndex].nodes[nodeIndex].Weights[weightIndex];
                    
                //cout << currentLayerIndex-1 << " " << nodeIndex << " " << weightIndex << endl;
                //cout << prevLayerData << " ";
                //cout << currentLayerWeight << endl;

                preSigmoidTotal += (prevLayerData * currentLayerWeight);
            }
            preSigmoidTotal += layers[currentLayerIndex].nodes[nodeIndex].Bias;
            layers[currentLayerIndex][nodeIndex].preSigmoidData = preSigmoidTotal;

            layers[currentLayerIndex].nodes[nodeIndex].data = Sigmoid(preSigmoidTotal);
            preSigmoidTotal = 0;
        }
    }
}

//Gets strongest node
int Network::getAnswer() {
    int largestEndNode = 0;
    for (int i = 1; i < layers[networkSize- 1].size; i++) {
        if ((*outputLayer)[i].data > (*outputLayer)[largestEndNode].data) {
            largestEndNode = i;
        }
    }
    return largestEndNode;
}

//Gets if the network correctly guessed
bool Network::isAnswerCorrect(vector<double> expectedAnswers) {
    int answerIndex = 0;
    for (int i = 1; i < layers[networkSize- 1].size; i++) {
        if (expectedAnswers[i] > expectedAnswers[answerIndex]) {
            answerIndex = i;
        }
    }

    if (getAnswer() == answerIndex) {
        return 1;
    }
    return 0;
}

//Gets the output layer
vector<double> Network::getAnswerVector() {
    vector<double> answerVector{};
    for (int i = 0; i < (*outputLayer).size; i++) {
        answerVector.push_back((*outputLayer)[i].data);
    }

    return answerVector;
}

double Network::getCost(vector<double> expectedAnswers) {
    if (expectedAnswers.size() != (*outputLayer).size) {
        throw std::out_of_range("More answers than expected");
    }

    double cost = 0;
    vector<double> actualAnswers = getAnswerVector();

    for (int i = 0; i < (*outputLayer).size; i++) {
        cost += pow( (actualAnswers[i] - expectedAnswers[i]) , 2);
    }

    return cost;
}

vector<vector<vector<double>>> Network::differentiate(vector<double> expectedAnswers) {
    //how the network would like the node data to change, used to calculate weight and bias gradients
    vector<vector<double>> nodeGradientVector(networkSize);
    //gradient of the controlable values (weights, biases)
    vector<vector<vector<double>>> controlableGradients(networkSize);
    for (int i = 0; i < networkSize; i++) {
        controlableGradients[i] = vector<vector<double>>(structure[i]);
    }

    //calculates input layer gradients
    for (int i = 0; i < structure[networkSize-1]; i++) {
        double nodeGradient = 2 * (layers[networkSize-1][i].data - expectedAnswers[i]);
        nodeGradientVector[networkSize-1].push_back(nodeGradient);
    }

    //for the amount of hidden layers
    for (int i = networkSize-2; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (int j = 0; j < structure[i]; j++) {
            double nodeGradient = getNodeGradient(nodeGradientVector[i+1], j, i);
            nodeGradientVector[i].push_back(nodeGradient);
        }
    }

    //for the amount of layers with data
    for (int i = networkSize-1; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (int j = 0; j < structure[i]; j++) {
            //for the amount of weights in the current node
            for (int k = 0; k < structure[i-1]; k++) {
                double weightGradient = getGradient(nodeGradientVector[i][j], i, j, k);
                controlableGradients[i][j].push_back(weightGradient);
            }

            double biasGradient = getGradient(nodeGradientVector[i][j], i, j);
            controlableGradients[i][j].push_back(biasGradient);
        }
    }
        
    return controlableGradients;
}

//input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double Network::getNodeGradient(vector<double> nextLayerNodeGradients, int nodePos, int currentLayerPos) {
    double nodeGradient = 0.0;

    double currentLayerNodeWeight = 0.0;
    double preSigmoidData = 0.0;
    double derivativeAjToAk = 0.0;

    //  nL-1
    //  \    dC0   daj
    //  /    --- x ---
    //  j=0  daj   dak
    for (int j = 0; j < nextLayerNodeGradients.size(); j++) {

        //z
        preSigmoidData = layers[currentLayerPos + 1][j].preSigmoidData;

        //wjk
        currentLayerNodeWeight = layers[currentLayerPos + 1][j][nodePos];

        // daj
        // ---
        // dak
        derivativeAjToAk = SigmoidDerivative(preSigmoidData) * currentLayerNodeWeight;

        // dC0   daj
        // --- x ---
        // daj   dak
        nodeGradient += nextLayerNodeGradients[j] * derivativeAjToAk;
    }

    return nodeGradient;
}

//input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
double Network::getGradient(double nextLayerNodeGradient, int layerPos, int nodePos, int weightPos) {

    double preSigmoidNodeData = layers[layerPos][nodePos].preSigmoidData;

    //The data that the weight is multiplied by is park of the derivative
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

double Network::getGradient(double nextLayerNodeGradient, int layerPos, int nodePos) {

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

tuple< vector<vector<vector<double>>>, double, bool> Network::Test(vector<vector<double>> testExample) {
    input(testExample[0]);
    calculate();

    vector<vector<vector<double>>> gradient = differentiate(testExample[1]);
    double cost = getCost(testExample[1]);

    bool isCorrect = false;
    if (isAnswerCorrect(testExample[1])) {
        isCorrect = true;
    }

    return {gradient, cost, isCorrect};
}

tuple< vector<vector<vector<double>>>, double, double> Network::TestSet(vector<vector<vector<double>>> testSet) {
    double averageCorrect = 0.0;
    double averageCost = 0.0;
    vector<vector<vector<double>>> averageGradient(networkSize);
        
    for (int i = 1; i < networkSize; i++) {
        averageGradient[i] = vector<vector<double>>(structure[i]);
        for(int j = 0; j < structure[i]; j++) {
            averageGradient[i][j] = vector<double>(structure[i-1]+1);
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
        for (int i = networkSize-1; i > 0; i--) {
            //for the amount of nodes in the current layer
            for (int j = 0; j < structure[i]; j++) {
                //for the amount of weights in the current node
                for (int k = 0; k < structure[i-1] + 1; k++) {
                    averageGradient[i][j][k] += gradient[i][j][k];
                }
            }
        }
    }

    //average gradient sum
    for (int i = networkSize-1; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (int j = 0; j < structure[i]; j++) {
            //for the amount of weights in the current node
            for (int k = 0; k < structure[i-1] + 1; k++) {
                averageGradient[i][j][k] /= testSet.size();
            }
        }
    }
    averageCost /= testSet.size();
    averageCorrect /= testSet.size();
        

    return {averageGradient, averageCost, averageCorrect};
}
    
//removes gradient from the weights and biases
void Network::alterByGradient(vector<vector<vector<double>>> averageGradient, double learningRate) {
    //for the amount of layers with data
    for (int i = networkSize-1; i > 0; i--) {
        //for the amount of nodes in the current layer
        for (int j = 0; j < structure[i]; j++) {
            //for the amount of weights in the current node
            for (int k = 0; k < structure[i-1]; k++) {

                layers[i][j][k] -= averageGradient[i][j][k] * learningRate;
                //cout << i << " " << j << " " << k << endl;
                //cout << layers[i][j][k] << " " << gradientSum[i][j][k] * learningRate << endl;
            }
            //cout << i << " " << j << " bias" << endl;
            layers[i][j].Bias -= averageGradient[i][j][structure[i]-1] * learningRate;
            //cout << layers[i][j].Bias << " " << gradientSum[i][j][structure[i]-1] * learningRate << endl;
        }
    }
}

void Network::improveNetworkBackPropogation(vector<vector<vector<double>>> testSet, double learningRate) {

    auto averageTestResult = TestSet(testSet);

    vector<vector<vector<double>>> averageGradient = get<0>(averageTestResult);
    double averageCost = get<1>(averageTestResult);
    double averageCorrect = get<2>(averageTestResult);

    alterByGradient(averageGradient, learningRate);

    cout << "percentage correct: " << averageCorrect << endl;
    cout << "cost is: " << averageCost << endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////