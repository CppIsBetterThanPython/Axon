#pragma once

#include <filesystem>
#include <optional>

#include "utils.h"

// TODO: Dynamically split the 1d vector
// Structure to store basic parameters of the network
struct Parameters {
private:
    std::vector<double> weightsData;
    std::vector<double> biasesData;
public:
    std::vector<std::vector<std::span<double>>> weights;
    std::vector<std::span<double>> biases;
    size_t size;
    std::vector<size_t> structure;

    friend bool saveParameters(const Parameters& parameters, const std::filesystem::path& filePath);

    friend Parameters getParameters(const std::filesystem::path& filePath);

    Parameters(std::vector<size_t> structure) {
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

    Parameters(Parameters&& other) noexcept
        : weightsData(std::move(other.weightsData)),
        biasesData(std::move(other.biasesData)),
        structure(std::move(other.structure)),
        size(std::move(other.size)) {

        moveSpans();
    }

    Parameters(const Parameters& other)
        : weightsData(other.weightsData),
        biasesData(other.biasesData),
        structure(other.structure),
        size(other.size) {

        moveSpans();
    }

    Parameters& operator=(const Parameters& other) {
        if (this == &other)
            return *this;

        weightsData = other.weightsData;
        biasesData = other.biasesData;
        structure = other.structure;
        size = other.size;

        moveSpans();

        return *this;
    }

    void initParameters() {
        for (size_t layerID = 0; layerID < size; layerID++)
            for (std::span<double>& node : weights[layerID])
                for (double& weight : node)
                    weight = XavierInitialization(structure[layerID], structure[layerID + 1]);

        for (double& bias : biasesData)
            bias = 0.1;
    }

    // TODO: Evaluate if clear is neccessary
    void moveSpans() {
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

    Parameters operator+(Parameters other) const {
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

    Parameters operator-(Parameters other) const {
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

    template<typename T>
    Parameters operator*(T scalar) const {
        Parameters scaled = Parameters(structure);

        for (int i = 0; i < weightsData.size(); i++) {
            scaled.weightsData[i] = this->weightsData[i] *  scalar;
        }

        for (int i = 0; i < biasesData.size(); i++) {
            scaled.biasesData[i] = this->weightsData[i] * scalar;
        }

        return scaled;
    }

    template<typename T>
    Parameters operator/(T scalar) const {
        Parameters scaled = Parameters(structure);

        for (int i = 0; i < weightsData.size(); i++) {
            scaled.weightsData[i] = this->weightsData[i] / scalar;
        }

        for (int i = 0; i < biasesData.size(); i++) {
            scaled.biasesData[i] = this->biasesData[i] / scalar;
        }

        return scaled;
    }

    bool operator==(const Parameters& other) const {
        if (other.structure != structure)
            return false;

        if (other.weightsData != weightsData || other.biasesData != biasesData)
            return false;

        return true;
    }
};

class NetworkBase {
    std::vector<size_t> structure;
    size_t netSize;
public:

    NetworkBase(const std::vector<size_t>& structure) : structure(structure) {
        netSize = structure.size();
    }

    virtual void input(const std::vector<std::vector<double>>&) = 0;
    virtual void input(const std::vector<double>& input) = 0;
    virtual void calculate() = 0;
    virtual std::vector<double> getAnswerVector() = 0;
    virtual std::vector<std::vector<double>> getAnswerVectors() = 0;

    // Gets strongest node
    virtual size_t getAnswer() {
        std::vector<double> answer = getAnswerVector();
        return getLargestID( answer );
    }

    // Gets if the network correctly guessed
    virtual bool isAnswerCorrect(const std::vector<double>& expectedAnswers) { return (getLargestID(expectedAnswers) == getAnswer()); }

    inline const size_t& size()            const { return netSize;           }
    inline const size_t& inputLayerSize()  const { return structure.front(); }
    inline const size_t& outputLayerSize() const { return structure.back();  }

    inline const std::vector<size_t>& getStructure()             const { return structure;        }
    inline const size_t&              getStructure(size_t index) const { return structure[index]; }
};

class GPU;

namespace cl {
    class Buffer;
}

class NetworkCPU : public NetworkBase {
protected:
    Parameters& parameters;
    std::vector<double> nodeDataRaw;
    std::vector<std::span<double>> nodeData;

    std::vector<std::vector<double>> batchedInputs;
    std::vector<std::vector<double>> batchedOutputs;
public:

    NetworkCPU(Parameters& parameters);

    void input(const std::vector<std::vector<double>>&) override;
    void input(const std::vector<double>&) override;

    void calculate() override;
private:
    void calculatePass();
public:

    std::vector<double> getAnswerVector() override;
    std::vector<std::vector<double>> getAnswerVectors() override;
};

class NetworkGPU : public NetworkBase {
protected:
    std::vector<std::unique_ptr<cl::Buffer>> WeightBuffers;
    std::vector<std::unique_ptr<cl::Buffer>> BiasBuffers;

    size_t batchSize = 1;

    std::unique_ptr<cl::Buffer> inputBuffer;
    std::unique_ptr<cl::Buffer> outputBuffer;

    std::unique_ptr<GPU> gpu;

    Parameters& parameters;
public:

    NetworkGPU(Parameters& parameters);

    ~NetworkGPU();

    void input(const std::vector<std::vector<double>>&) override;
    void input(const std::vector<double>&) override;
    void calculate() override;
    std::vector<double> getAnswerVector() override;
    std::vector<std::vector<double>> getAnswerVectors() override;

    void loadBuffers();
    void saveBuffers();

    inline size_t largestLayer() const {
        size_t curLargestLayer = 0;

        for (size_t i = 0; i < size(); i++)
            curLargestLayer = std::max(curLargestLayer, getStructure(i));

        return curLargestLayer;
    }

    inline size_t largestPassLayer() const {
        size_t curLargestLayer = 0;

        for (size_t i = 1; i < size(); i++)
            curLargestLayer = std::max(curLargestLayer, getStructure(i));

        return curLargestLayer;
    }
};

// TODO: Add enum state and limit what functions can be called based on state.
// TODO: Add a optimising option, which allows for the class to decide when to use which interface.
class Network : public NetworkBase {
public:
    enum class Interface { CPU, GPU };
    enum class State { Ready, Inputted, Calculated };
    enum class InputType { Singular, Batched };
protected:
    Parameters parameters;

    // Unique pointer is for polymorphism for other network types.
    std::optional<std::unique_ptr<NetworkCPU>> cpuInterface;
    std::optional<std::unique_ptr<NetworkGPU>> gpuInterface;

    Interface interface_;
    State state;
    InputType inputType;

    // initParameters is to defer initialisation to derived classes to avoid initialising twice.
    Network(const Parameters& parameters, Interface interface_ = Interface::GPU, bool initParameters = true);
public:
    static std::unique_ptr<Network> createNetwork(const Parameters& parameters, Interface interface_ = Interface::GPU);
    static std::unique_ptr<Network> createNetwork(const std::vector<size_t>& Structure, Interface interface_ = Interface::GPU);
    static std::unique_ptr<Network> createNetwork(const std::filesystem::path& filename, Interface interface_ = Interface::GPU);

    virtual void switchInterface();

    ~Network ();

    virtual bool saveNetwork ( const std::filesystem::path & filename ) const;
    virtual bool loadNetwork ( const std::filesystem::path & filename );

    void input(const std::vector<std::vector<double>>&);
    void input (const std::vector<double>& input ) override;
    void calculate () override;
    std::vector<double> getAnswerVector() override;
    std::vector<std::vector<double>> getAnswerVectors() override;
};

Parameters getParameters(const std::filesystem::path& filePath);
bool saveParameters(const Parameters& parameters, const std::filesystem::path& filePath);