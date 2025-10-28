#include "NeuralNet.h"

Parameters::Parameters(std::vector<size_t> structure) {
    isInitialised = false;

    // TODO: Use std::accumulate
    size = structure.size() - 1;
    this->structure = structure;
    size_t weightsSize = 0;
    size_t biasesSize = 0;
    for (size_t i = 0; i < size; i++) {
        weightsSize += structure[i] * structure[i + 1];
        biasesSize += structure[i + 1];
    }
    weightsData = std::vector<double>(weightsSize);
    biasesData = std::vector<double>(biasesSize);

    weights.reserve(size);
    biases.reserve(size);

    size_t currentBiasIndexStart = 0;
    size_t currentWeightIndexStart = 0;
    for (size_t layerID = 0; layerID < size; layerID++) {
        const size_t prevLayerSize = structure[layerID];
        const size_t currentLayerSize = structure[layerID + 1];

        weights.push_back(std::vector<std::span<double>>(currentLayerSize));
        biases.push_back(std::span<double>(&biasesData[currentBiasIndexStart], currentLayerSize));
        for (std::span<double>& node : weights[layerID]) {
            node = std::span<double>(&weightsData[currentWeightIndexStart], prevLayerSize);

            currentWeightIndexStart += prevLayerSize;
        }

        currentBiasIndexStart += currentLayerSize;
    }
}

Parameters::Parameters(Parameters&& other) noexcept
    : weightsData(std::move(other.weightsData)),
    biasesData(std::move(other.biasesData)),
    structure(std::move(other.structure)),
    size(std::move(other.size)),
    isInitialised(std::move(other.isInitialised)) {

    moveSpans();
}

Parameters::Parameters(const Parameters& other)
    : weightsData(other.weightsData),
    biasesData(other.biasesData),
    structure(other.structure),
    size(other.size),
    isInitialised(other.isInitialised) {

    moveSpans();
}

Parameters& Parameters::operator=(const Parameters& other) {
    if (this == &other)
        return *this;

    weightsData = other.weightsData;
    biasesData = other.biasesData;
    structure = other.structure;
    size = other.size;
    isInitialised = other.isInitialised;

    moveSpans();

    return *this;
}

// TODO: Evaluate if clear is neccessary
void Parameters::moveSpans() {
    weights.clear();
    biases.clear();

    weights.reserve(size);
    biases.reserve(size);

    size_t currentBiasIndexStart = 0;
    size_t currentWeightIndexStart = 0;
    for (size_t layerID = 0; layerID < size; layerID++) {
        const size_t prevLayerSize = structure[layerID];
        const size_t currentLayerSize = structure[layerID + 1];

        weights.push_back(std::vector<std::span<double>>(currentLayerSize));
        biases.push_back(std::span<double>(&biasesData[currentBiasIndexStart], currentLayerSize));
        for (std::span<double>& node : weights[layerID]) {
            node = std::span<double>(&weightsData[currentWeightIndexStart], prevLayerSize);

            currentWeightIndexStart += prevLayerSize;
        }

        currentBiasIndexStart += currentLayerSize;
    }
}

Parameters Parameters::operator+(Parameters other) const {
    if (other.structure != structure)
        throw std::invalid_argument("class Parameters: Parameters addition is only valid for parameters of the same structure.");

    Parameters added = Parameters(structure);

    for (int i = 0; i < weightsData.size(); i++) {
        added.weightsData[i] = this->weightsData[i] + other.weightsData[i];
    }

    for (int i = 0; i < biasesData.size(); i++) {
        added.biasesData[i] = this->biasesData[i] + other.biasesData[i];
    }

    return added;
}

Parameters Parameters::operator-(Parameters other) const {
    if (other.structure != structure)
        throw std::invalid_argument("class Parameters: Parameters addition is only valid for parameters of the same structure.");

    Parameters subtracted = Parameters(structure);

    for (int i = 0; i < weightsData.size(); i++) {
        subtracted.weightsData[i] = this->weightsData[i] - other.weightsData[i];
    }

    for (int i = 0; i < biasesData.size(); i++) {
        subtracted.biasesData[i] = this->biasesData[i] - other.biasesData[i];
    }

    return subtracted;
}

bool Parameters::operator==(const Parameters& other) const {
    if (other.structure != structure)
        return false;

    if (other.weightsData != weightsData || other.biasesData != biasesData)
        return false;

    return true;
}