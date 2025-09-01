#include <cstdlib> // For random numbers
#include <ctime> // So is this

#include "NeuralNet.h"
#include "GPU.h"

using std::vector, std::tuple;

// Node and Layer classes defined in header

void Network::initGPU() {
    gpu = std::make_unique<GPU>();

    WeightBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
    BiasBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
    saveBuffers();
}

Network::Network(const vector<size_t>& Structure, bool useGPU) {
    netSize = Structure.size();
    structure = Structure;
    // Reserves enough data for the layers
    layers.reserve(size());

    for(size_t i = 0; i < size(); i++) {
        layers.push_back(Layer(structure[i], (i > 0) ? structure[i - 1] : 0));
    }

    inputLayer = &layers[0];
    outputLayer = &layers[size() - 1];

    GPUacceleration = useGPU;
    if (GPUacceleration) {
        gpu = std::make_unique<GPU>();

        WeightBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        BiasBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        saveBuffers();
    }
}

Network::Network(const Parameters& parameters, bool useGPU) {
    this->SetParameters(parameters);

    GPUacceleration = useGPU;
    if (GPUacceleration) {
        gpu = std::make_unique<GPU>();
        WeightBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        BiasBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        saveBuffers();
    }
}

Network::Network(const std::filesystem::path& filePath, bool useGPU) {
    Parameters parameters;

    if (this->getNetwork(parameters, filePath)) {
        throw std::ios_base::failure("Failed to open file for writing: " + filePath.string());
    }

    netSize = parameters.structure.size();
    structure = parameters.structure;
    // Reserves enough data for the layers
    layers.reserve(size());

    for (size_t i = 0; i < netSize; i++) {
        layers.push_back(Layer(structure[i], (i > 0) ? structure[i - 1] : 0));
    }

    this->SetParameters(parameters);

    inputLayer = &layers[0];
    outputLayer = &layers[netSize - 1];

    GPUacceleration = useGPU;
    if (GPUacceleration) {
        WeightBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        BiasBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
        gpu = std::make_unique<GPU>();
        saveBuffers();
    }
}

// TODO: Better handle back ups
Network::~Network() {};

void Network::resetWeights() {
    layers.clear();

    for (size_t i = 0; i < size(); i++) {
        layers[i] = Layer(structure[i], (i > 0) ? structure[i - 1] : 0);
    }
}

// Input vector to set input node layer
void Network::input(vector<double> input) {
    if (input.size() != inputLayer->size())
        throw std::out_of_range("Wrong input layer size");
    
    for (int i = 0; i < inputLayer->size(); i++)
        (*inputLayer)[i].data = input[i];
}

// Passes data through layers
void Network::calculate() {
    if (GPUacceleration) {
        if (gpu) {
            calculateGPU();
            return;
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    calculateCPU();
}

// Passes data through layers
void Network::calculateCPU() {
    for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {

        for (size_t nodeIndex = 0; nodeIndex < layers[currentLayerIndex].size(); nodeIndex++) {
            double preSigmoidTotal = 0;

            for (size_t weightIndex = 0; weightIndex < layers[currentLayerIndex - 1].size(); weightIndex++) {
                //multiply previous nodes data by it's corresponding weight in the current node

                double prevLayerData = layers[currentLayerIndex - 1][weightIndex].data;
                double currentLayerWeight = layers[currentLayerIndex][nodeIndex][weightIndex];

                preSigmoidTotal += (prevLayerData * currentLayerWeight);
            }
            preSigmoidTotal += layers[currentLayerIndex][nodeIndex].bias;

            layers[currentLayerIndex][nodeIndex].data = Sigmoid(preSigmoidTotal);
        }
    }
}

// Gets the output layer
vector<double> Network::getAnswerVector() {
    vector<double> answerVector{};

    for (int i = 0; i < outputLayer->size(); i++)
        answerVector.push_back((*outputLayer)[i].data);

    return answerVector;
}

// Gets strongest node
int Network::getAnswer() {
    int largestEndNode = 0;
    for (int i = 1; i < layers[netSize- 1].size(); i++)
        if ((*outputLayer)[i].data > (*outputLayer)[largestEndNode].data)
            largestEndNode = i;
    
    return largestEndNode;
}

// Gets if the network correctly guessed
bool Network::isAnswerCorrect(vector<double> expectedAnswers) {
    int answerIndex = 0;
    for (int i = 1; i < layers[netSize- 1].size(); i++)
        if (expectedAnswers[i] > expectedAnswers[answerIndex])
            answerIndex = i;

    if (getAnswer() == answerIndex)
        return 1;

    return 0;
}

void Network::loadBuffers() {
    if (gpu) {
        // TODO: implement
    }
}

void Network::saveBuffers() {
    if (gpu) {
        //WeightBuffers.value().empty();
        WeightBuffers.value().reserve(size());

        //BiasBuffers.value().empty();
        BiasBuffers.value().reserve(size());

        for (size_t i = 1; i < size(); i++) {
            Layer& layer = layers[i];
            vector<double> layerWeights;
            vector<double> layerBiases;
            // TODO: reserve size
            for (const Node& node : layer.nodes) {
                layerWeights.insert(layerWeights.end(), node.Weights.begin(), node.Weights.end());
                layerBiases.push_back(node.bias);
            }
            cl::Buffer weightBuffer(gpu.value()->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerWeights.size(), layerWeights.data());
            cl::Buffer biasBuffer(gpu.value()->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerBiases.size(), layerBiases.data());

            WeightBuffers.value().push_back(std::make_unique<cl::Buffer>(weightBuffer));
            BiasBuffers.value().push_back(std::make_unique<cl::Buffer>(biasBuffer));
        }
    }
}

void Network::SetParameters(Parameters parameters) {
    if (this->netSize != parameters.size + 1)
        throw std::out_of_range("Index out of bounds");
    else if (this->structure != parameters.structure)
        throw std::out_of_range("Incorrect structure");

    for (size_t layer = 0; layer < netSize - 1; layer++) {
        for (size_t node = 0; node < structure[layer + 1]; node++) {

            for (size_t weight = 0; weight < structure[layer]; weight++)
                layers[layer + 1][node][weight] = parameters.data[layer][node][weight];

            layers[layer + 1][node].bias = parameters.data[layer][node].back();
        }
    }
}

Parameters Network::GetParameters() const {
    Parameters parameters;
    parameters.size = size() - 1;
    parameters.structure = structure;
    // TODO: Make use reserve and not constructor
    parameters.data = vector<vector<vector<double>>>(parameters.size);

    for (size_t layer = 0; layer < parameters.size; layer++) {

        parameters.data[layer] = vector<vector<double>>(structure[layer + 1]);

        for (size_t node = 0; node < structure[layer + 1]; node++) {
            parameters.data[layer][node] = vector<double>(structure[layer] + 1);

            for (size_t weight = 0; weight < structure[layer]; weight++)
                parameters.data[layer][node][weight] = layers[layer + 1][node][weight];

            parameters.data[layer][node][structure[layer]] = layers[layer + 1][node].bias;
        }
    }
    return parameters;
}

Parameters Network::EmptyParameters() {
    Parameters parameters;
    parameters.size = size() - 1;
    parameters.structure = structure;
    parameters.data = vector<vector<vector<double>>>(parameters.size);
    for (size_t i = 0; i < parameters.size; i++) {
        parameters.data[i] = vector<vector<double>>(structure[i + 1]);
        for (size_t j = 0; j < structure[i + 1]; j++) {
            parameters.data[i][j] = vector<double>(structure[i] + 1, 0);
        }
    }

    return parameters;
}