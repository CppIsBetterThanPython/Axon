#pragma once

#include <filesystem>
#include <optional>

#include "pch.h"

// To turn values into 0-1 range
static inline double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

// For random number to use in Xavier Initialization
inline double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

bool initOpenCL();

std::vector<double> multiplyVectorOpenCL(const std::vector<double>& input, double factor);

// More efficient and stable initialisation
static inline double XavierInitialization(size_t in, size_t out) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal() * range;  // Scaled random value
    return x;
}

// Structure to store basic parameters of the network
struct Parameters {
    std::vector<std::vector<std::vector<double>>> data = {};
    size_t size = 0;
    std::vector<size_t> structure = {};

    Parameters() : data({}), size(0), structure({}) {}

    Parameters(std::vector<size_t> structure) {
        // TODO: Use reserve.
        this->size = structure.size() - 1;
        this->structure = structure;
        this->data = std::vector<std::vector<std::vector<double>>>(this->size);

        for (size_t i = 0; i < this->size; i++) {
            this->data[i] = std::vector<std::vector<double>>(structure[i + 1]);
            for (int j = 0; j < structure[i + 1]; j++) {
                this->data[i][j] = std::vector<double>(structure[i] + 1);
                for (double& parameter : this->data[i][j])
                    parameter = 0;
            }
        }
    }

    inline std::vector<std::vector<double>>& operator[](size_t index) {
        if (index > size)
            throw std::out_of_range("Index out of bounds");

        return data[index];
    }

    inline const std::vector<std::vector<double>>& operator[](size_t index) const {
        if (index > size)
            throw std::out_of_range("Index out of bounds");

        return data[index];
    }
};

class Network;

class Node {
private:
    friend class Network;

    std::vector<double> Weights = {};
public:
    double& data,
            bias;

    Node(double& data, double& bias, const size_t prevLayerSize, const size_t curLayerSize) : data(data), bias(bias) {
        Weights.reserve(prevLayerSize);
        for (size_t i = 0; i < prevLayerSize; i++)
            Weights.push_back(XavierInitialization(prevLayerSize, curLayerSize));
    }

    operator std::vector<double>() const {
        std::vector<double> controlables = Weights;  // Copy Weights into a new vector
        controlables.push_back(bias);  // Add Bias to the vector
        return controlables;  // Return the result
    }

    operator double() { return data; }

    double& operator[](size_t index) {
        if (index >= Weights.size())
            throw std::out_of_range("Index out of bounds");

        return Weights[index];
    }

    const double operator[](size_t index) const {
        if (index >= Weights.size())
            throw std::out_of_range("Index out of bounds");

        return Weights[index];
    }
};

class Layer {
private:
    friend class Node;
    friend class Network;
    size_t         layerSize = 0;
    std::vector<Node>   nodes = {};
protected:
    std::vector<double> biases = {};
    std::vector<double> nodeData = {};
public:
    // TODO: Make biases and data single vector in layer

    Layer() : layerSize(0), nodes({}) {}

    Layer(size_t size, size_t prevLayerSize) {
        this->layerSize = size;

        nodes.reserve(size);
        biases.reserve(size);
        nodeData.reserve(size);
        for (size_t i = 0; i < size; i++) {
            nodeData.push_back(0.0);
            biases.push_back(0.1);
            nodes.push_back(Node{ nodeData[i], biases[i], prevLayerSize, size});
        }
    }

    inline const size_t size() const {
        return layerSize;
    }

    operator std::vector<Node>() { return nodes; }

    operator std::vector<double>& () {
        std::vector<double> dataVect = {};
        for (double data : nodes)
            dataVect.push_back(data);

        return dataVect;
    }

    inline Node& operator[](size_t index) {
        if (index >= size())
            throw std::out_of_range("Index out of bounds");

        return nodes[index];
    }

    const Node& operator[](size_t index) const {
        if (index >= size())
            throw std::out_of_range("Index out of bounds");

        return nodes[index];
    }
};

class GPU;

namespace cl {
    class Buffer;
}

class Network {
private:
    std::optional<std::vector<std::unique_ptr<cl::Buffer>>> WeightBuffers;
    std::optional<std::vector<std::unique_ptr<cl::Buffer>>> BiasBuffers;
    std::optional<std::unique_ptr<GPU>> gpu;
    bool GPUacceleration;
    // layers[layer][node][weight]
    std::vector<Layer> layers;
    size_t netSize;
public:
    Layer* inputLayer;
    Layer* outputLayer;
    std::vector<size_t> structure;
private:
    void initGPU();
public:
    Network (const std::vector<size_t>& Structure, bool useGPU = true);
    Network (const Parameters& parameters, bool useGPU = true);
    Network (const std::filesystem::path& filename, bool useGPU = true);

    ~Network ();

    inline const size_t size() const {
        return netSize;
    }

    inline const size_t largestLayer() const {
        size_t curLargestLayer = 0;

        for (Layer layer : layers) {
            curLargestLayer = std::max(curLargestLayer, layer.size());
        }

        return curLargestLayer;
    }

    inline const size_t largestPassLayer() const {
        size_t curLargestLayer = 0;

        for (size_t i = 1; i < size(); i++) {
            curLargestLayer = std::max(curLargestLayer, layers[i].size());
        }

        return curLargestLayer;
    }

    operator std::vector<Layer>() { return layers; }

    operator Parameters() { return this->GetParameters(); }

    inline Layer& operator[](size_t index) {
        if (index >= size())
            throw std::out_of_range("Index out of bounds");
        
        return layers[index];
    }

    inline const Layer& operator[](size_t index) const {
        if (index >= size())
            throw std::out_of_range("Index out of bounds");

        return layers[index];
    }

    void resetWeights ();

    bool getNetwork(Parameters& parameters, const std::filesystem::path& filename);
    bool saveNetwork ( const std::filesystem::path & filename ) const;
    bool loadNetwork ( const std::filesystem::path & filename );

    void input (std::vector<double> input );
    void calculate ();
    void calculateCPU ();
    void calculateGPU ();

    // Gets strongest node
    int getAnswer ();

    // Gets if the network correctly guessed
    bool isAnswerCorrect (std::vector<double> expectedAnswers );

    // Gets the output layer
    std::vector<double> getAnswerVector ();

protected:
    void loadBuffers();
    void saveBuffers();
public:

    void SetParameters ( Parameters parameters );
    Parameters GetParameters() const;

    Parameters EmptyParameters ();
};