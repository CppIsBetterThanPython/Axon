#include <cstdlib> // For random numbers
#include <ctime> // So is this

#include "pch.h"
#include "NeuralNet.h"

using std::vector, std::tuple;

// Node and Layer classes defined in header

//constructor
Network::Network(const vector<size_t>& Structure) {
    networkSize = Structure.size();
    structure = Structure;
    //defines amount of layers
    layers = vector<Layer>(networkSize);

    for(size_t i = 0; i < networkSize; i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i-1] : 0);
    }

    inputLayer = &layers[0];
    outputLayer = &layers[networkSize - 1];
}

Network::Network(const std::filesystem::path& filePath) {
    Parameters parameters;

    this->loadNetwork(filePath);

    networkSize = parameters.structure.size();
    structure = parameters.structure;
    //defines amount of layers
    layers = vector<Layer>(networkSize);

    for (size_t i = 0; i < networkSize; i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i - 1] : 0);
    }

    this->SetParameters(parameters);

    inputLayer = &layers[0];
    outputLayer = &layers[networkSize - 1];
}

Network::~Network() {
    std::string backupName = "BACKUP_";
    backupName += rand();
    this->saveNetwork(backupName);
}

void Network::resetWeights() {
    layers.clear();

    for (size_t i = 0; i < networkSize; i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i - 1] : 0);
    }
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
    // TODO: Add GPU acceleration
    for (size_t currentLayerIndex = 1; currentLayerIndex < networkSize; currentLayerIndex++) {

        for (size_t nodeIndex = 0; nodeIndex < layers[currentLayerIndex].size; nodeIndex++) {
            double preSigmoidTotal = 0;

            for (size_t weightIndex = 0; weightIndex < layers[currentLayerIndex - 1].size; weightIndex++) {
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
    if (this->networkSize != parameters.size + 1)
        throw std::out_of_range("Index out of bounds");
    else if (this->structure != parameters.structure)
        throw std::out_of_range("Incorrect structure");

    for (size_t layer = 0; layer < networkSize - 1; layer++) {
        for (size_t node = 0; node < structure[layer + 1]; node++) {

            for (size_t weight = 0; weight < structure[layer]; weight++)
                layers[layer + 1][node][weight] = parameters.data[layer][node][weight];

            layers[layer + 1][node].Bias = parameters.data[layer][node].back();
        }
    }
}

Parameters Network::GetParameters() const {
    Parameters parameters;
    parameters.size = networkSize - 1;
    parameters.structure = structure;
    parameters.data = vector<vector<vector<double>>>(parameters.size);

    for (size_t layer = 0; layer < parameters.size; layer++) {

        parameters.data[layer] = vector<vector<double>>(structure[layer + 1]);

        for (size_t node = 0; node < structure[layer + 1]; node++) {
            parameters.data[layer][node] = vector<double>(structure[layer] + 1);

            for (size_t weight = 0; weight < structure[layer]; weight++)
                parameters.data[layer][node][weight] = layers[layer + 1][node][weight];

            parameters.data[layer][node][structure[layer]] = layers[layer + 1][node].Bias;
        }
    }
    return parameters;
}

Parameters Network::EmptyParameters() {
    Parameters parameters;
    parameters.size = networkSize - 1;
    parameters.structure = structure;
    parameters.data = vector<vector<vector<double>>>(parameters.size);
    for (size_t i = 0; i < parameters.size; i++) {
        parameters.data[i] = vector<vector<double>>(structure[i + 1]);
        for (size_t j = 0; j < structure[i + 1]; j++) {
            parameters.data[i][j] = vector<double>(structure[i] + 1);
            for (double& parameter : parameters.data[i][j])
                parameter = 0;
        }
    }

    return parameters;
}