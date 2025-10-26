#include "NeuralNet.h"

NetworkCPU::NetworkCPU(Parameters& parameters) : NetworkBase(parameters.structure), parameters(parameters) {
    
    size_t nodeDataSize = std::accumulate(getStructure().begin(), getStructure().end(), 0);
    nodeDataRaw = std::vector<double>(nodeDataSize, 0);

    nodeData.reserve(size());

    size_t currentNodeIndexStart = 0;
    for (size_t layerID = 0; layerID < size(); layerID++) {
        const size_t currentLayerSize = getStructure()[layerID];

        nodeData.push_back(std::span<double>(&nodeDataRaw[currentNodeIndexStart], currentLayerSize));

        currentNodeIndexStart += currentLayerSize;
    }
}

// Batched Input
void NetworkCPU::input(const std::vector<std::vector<double>>& input) {
    if (input.size() == 0)
        throw std::invalid_argument("NetworkCPU::input: Cannot have 0 batches in an input.");

    for (const std::vector<double>& batch : input)
        if (batch.size() != inputLayerSize())
            throw std::invalid_argument("NetworkCPU::input: Input size does not match input layer size. Expected: " + std::to_string(inputLayerSize()) +
                " but got:" + std::to_string(batch.size()));

    batchedInputs = input;
}

// TODO: maybe make this just add one to batchedInputs and make another function to load inputs
void NetworkCPU::input(const std::vector<double>& input) {
    if (input.size() != getStructure().front())
        throw std::out_of_range("Wrong input layer size");

    for (size_t i = 0; i < getStructure().front(); i++)
        nodeData.front()[i] = input[i];
}

// TODO: make check for input
void NetworkCPU::calculate() {
    if (batchedInputs.size() == 0) {
        calculatePass();
        return;
    }

    batchedOutputs = {};
    batchedOutputs.reserve(batchedInputs.size());

    for (std::vector<double> input : batchedInputs) {
        this->input(input);

        calculatePass();

        batchedOutputs.push_back(getAnswerVector());
    }
    batchedOutputs.shrink_to_fit();
}

void NetworkCPU::calculatePass() {
    for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {

        for (size_t nodeIndex = 0; nodeIndex < getStructure()[currentLayerIndex]; nodeIndex++) {
            double preSigmoidTotal = 0;

            for (size_t weightIndex = 0; weightIndex < getStructure()[currentLayerIndex - 1]; weightIndex++) {
                //multiply previous nodes data by it's corresponding weight in the current node

                double prevLayerData = nodeData[currentLayerIndex - 1][weightIndex];
                double currentLayerWeight = parameters.weights[currentLayerIndex - 1][nodeIndex][weightIndex];

                preSigmoidTotal += (prevLayerData * currentLayerWeight);
            }
            preSigmoidTotal += parameters.biases[currentLayerIndex - 1][nodeIndex];

            nodeData[currentLayerIndex][nodeIndex] = Sigmoid(preSigmoidTotal);
        }
    }
}

std::vector<double> NetworkCPU::getAnswerVector() const {
    std::vector<double> answerVector(nodeData.back().begin(), nodeData.back().end());

    return answerVector;
}

std::vector<std::vector<double>> NetworkCPU::getAnswerVectors() const {

    return batchedOutputs;
}