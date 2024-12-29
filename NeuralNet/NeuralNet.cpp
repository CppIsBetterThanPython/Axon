#include <cmath> // For basic math functions, such as pow

#include <cstdlib> // For random numbers
#include <ctime> // So is this

#include <vector> // (._.) < Really? ... )
#include <tuple> // For returning multiple, different type values
#include <iostream>

#include "NeuralNet.h"
#include "NN.h" // My filesaving class

using std::vector, std::tuple;

// Node and Layer classes defined in header

//constructor
Network::Network(vector<size_t> Structure) {
    networkSize = Structure.size();
    structure = Structure;
    //defines amount of layers
    layers = vector<Layer>(networkSize);

    for(int i = 0; i < networkSize; i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i-1] : 0);
    }

    inputLayer = &layers[0];
    outputLayer = &layers[networkSize - 1];
}

void Network::resetWeights() {
    layers.clear();

    for (int i = 0; i < networkSize; i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i - 1] : 0);
    }
}

void Network::saveNetwork(std::string filename) {
    NNFile networkFile = NNFile(filename);

    networkFile.write(*this);
}

void Network::loadNetwork(std::string filename) {
    NNFile networkFile = NNFile(filename);

    networkFile.read(*this);
}

// Input vector to set input node layer
void Network::input(vector<double> input) {
    if (input.size() != (*inputLayer).size)
        throw std::out_of_range("Wrong input layer size");
    
    for (int i = 0; i < (*inputLayer).size; i++)
        (*inputLayer)[i].data = input[i];
}

//passes data through layers
void Network::calculate() {
    for (int currentLayerIndex = 1; currentLayerIndex < networkSize; currentLayerIndex++) {

        for (int nodeIndex = 0; nodeIndex < layers[currentLayerIndex].size; nodeIndex++) {
            double preSigmoidTotal = 0;

            for (int weightIndex = 0; weightIndex < layers[currentLayerIndex - 1].size; weightIndex++) {
                //multiply previous nodes data by it's corresponding weight in the current node

                double prevLayerData = layers[currentLayerIndex - 1].nodes[weightIndex].data;
                double currentLayerWeight = layers[currentLayerIndex].nodes[nodeIndex].Weights[weightIndex];

                preSigmoidTotal += (prevLayerData * currentLayerWeight);
            }
            preSigmoidTotal += layers[currentLayerIndex].nodes[nodeIndex].Bias;
            layers[currentLayerIndex][nodeIndex].preSigmoidData = preSigmoidTotal;

            layers[currentLayerIndex].nodes[nodeIndex].data = Sigmoid(preSigmoidTotal);
        }
    }
}

// Gets the output layer
vector<double> Network::getAnswerVector() {
    vector<double> answerVector{};

    for (int i = 0; i < (*outputLayer).size; i++)
        answerVector.push_back((*outputLayer)[i].data);

    return answerVector;
}

// Gets strongest node
int Network::getAnswer() {
    int largestEndNode = 0;
    for (int i = 1; i < layers[networkSize- 1].size; i++)
        if ((*outputLayer)[i].data > (*outputLayer)[largestEndNode].data)
            largestEndNode = i;
    
    return largestEndNode;
}

// Gets if the network correctly guessed
bool Network::isAnswerCorrect(vector<double> expectedAnswers) {
    int answerIndex = 0;
    for (int i = 1; i < layers[networkSize- 1].size; i++)
        if (expectedAnswers[i] > expectedAnswers[answerIndex])
            answerIndex = i;

    if (getAnswer() == answerIndex)
        return 1;

    return 0;
}

void Network::SetParameters(Parameters parameters) {
    if (this->networkSize != parameters.size)
        throw std::out_of_range("Index out of bounds");
    else if (this->structure != parameters.structure)
        throw std::out_of_range("Incorrect structure");

    for (int layer = 0; layer < networkSize; layer++) {
        for (int node = 0; node < structure[layer + 1]; node++) {

            for (int weight = 0; weight < structure[node]; weight++)
                layers[layer + 1][node][weight] = parameters.data[layer][node][weight];

            layers[layer + 1][node].Bias = parameters.data[layer][node].back();
        }
    }
}

Network::Parameters Network::GetParameters() {
    Parameters parameters;
    parameters.size = networkSize - 1;
    parameters.structure = structure;
    parameters.data = vector<vector<vector<double>>>(parameters.size);
    for (int layer = 0; layer < parameters.size; layer++) {

        parameters.data[layer] = vector<vector<double>>(structure[layer + 1]);

        for (int node = 0; node < structure[layer + 1]; node++) {

            parameters.data[layer][node] = vector<double>(structure[node] + 1);

            for (int weight = 0; weight < structure[node]; weight++)
                parameters.data[layer][node][weight] = layers[layer][node][weight];

            parameters.data[layer][node].push_back(layers[layer][node].Bias);
        }
    }
    return parameters;
}

Network::Parameters Network::EmptyParameters() {
    Parameters parameters;
    parameters.size = networkSize - 1;
    parameters.structure = structure;
    parameters.data = vector<vector<vector<double>>>(parameters.size);
    for (int i = 0; i < parameters.size; i++) {
        parameters.data[i] = vector<vector<double>>(structure[i + 1]);
        for (int j = 0; j < structure[i + 1]; j++) {
            parameters.data[i][j] = vector<double>(structure[j] + 1);
            for (double& parameter : parameters.data[i][j]) {
                parameter = 0;
            }
        }
    }

    return parameters;
}