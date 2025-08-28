#pragma once

#include <vector>
#include <iostream>
#include <filesystem>

using std::vector, std::tuple;

// To turn values into 0-1 range
static inline double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

// For random number to use in Xavier Initialization
inline double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

// More efficient and stable initialisation
static inline double XavierInitialization(size_t in, size_t out) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal() * range;  // Scaled random value
    return x;
}

// Structure to store basic parameters of the network
struct Parameters {
    vector<vector<vector<double>>> data = {};
    size_t size = 0;
    vector<size_t> structure = {};

    Parameters() : data({}), size(0), structure({}) {}

    Parameters(vector<size_t> structure) {
        this->size = structure.size() - 1;
        this->structure = structure;
        this->data = vector<vector<vector<double>>>(this->size);

        for (int i = 0; i < this->size; i++) {
            this->data[i] = vector<vector<double>>(structure[i + 1]);
            for (int j = 0; j < structure[i + 1]; j++) {
                this->data[i][j] = vector<double>(structure[i] + 1);
                for (double& parameter : this->data[i][j])
                    parameter = 0;
            }
        }
    }

    inline vector<vector<double>>& operator[](size_t index) {
        if (index > size)
            throw std::out_of_range("Index out of bounds");

        return data[index];
    }

    inline const vector<vector<double>>& operator[](size_t index) const {
        if (index > size)
            throw std::out_of_range("Index out of bounds");

        return data[index];
    }
};

class Node {
public:
    double data = 0.0,
        Bias = 0.1,
        preSigmoidData = 0.0;

    vector<double> Weights = {};

    Node() : data(0.0), Weights({}), Bias(0.1), preSigmoidData(0.0) {}

    Node(size_t prevLayerNodes, size_t currentLayerNodes) : preSigmoidData(0.0), data(0.0), Bias(0.1) {
        Weights = vector<double>(prevLayerNodes);
        for (size_t i = 0; i < prevLayerNodes; i++) {
            Weights[i] = XavierInitialization(prevLayerNodes, currentLayerNodes);
        }
    }

    operator std::vector<double>() const {
        std::vector<double> controlables = Weights;  // Copy Weights into a new vector
        controlables.push_back(Bias);  // Add Bias to the vector
        return controlables;  // Return the result
    }

    operator double() { return data; }

    double& operator[](size_t index) {
        if (index >= Weights.size())
            throw std::out_of_range("Index out of bounds");

        return Weights[index];
    }

    const double& operator[](size_t index) const {
        if (index >= Weights.size())
            throw std::out_of_range("Index out of bounds");

        return Weights[index];
    }
};

class Layer {
public:
    size_t       size = 0;
    vector<Node> nodes = {};

    Layer() : size(0), nodes({}) {}

    Layer(size_t NodesAmount, size_t previousLayerNodes) {
        size = NodesAmount;
        nodes = vector<Node>(NodesAmount);
        for (int i = 0; i < NodesAmount; i++)
            nodes[i] = Node(previousLayerNodes, NodesAmount);
    }

    operator vector<Node>() { return nodes; }

    operator vector<double>& () {
        vector<double> dataVect = {};
        for (double data : nodes)
            dataVect.push_back(data);

        return dataVect;
    }

    Node& operator[](size_t index) {
        if (index >= size)
            throw std::out_of_range("Index out of bounds");

        return nodes[index];
    }

    const Node& operator[](size_t index) const {
        if (index >= size)
            throw std::out_of_range("Index out of bounds");

        return nodes[index];
    }
};

class Network {
protected:
    // layers[layer][node][weight]
    vector<Layer> layers;
    Layer* inputLayer;
    Layer* outputLayer;
public:
    size_t networkSize;
    vector<size_t> structure;

    Network (const vector<size_t>& Structure);
    
    Network(const std::filesystem::path& filename);

    ~Network ();

    operator vector<Layer>() { return layers; }

    operator Parameters() { return this->GetParameters(); }

    Layer& operator[](size_t index) {
        if (index >= networkSize)
            throw std::out_of_range("Index out of bounds");
        
        return layers[index];
    }

    void resetWeights ();

    bool getNetwork(Parameters& parameters, const std::filesystem::path& filename);
    bool saveNetwork ( const std::filesystem::path & filename ) const;
    bool loadNetwork ( const std::filesystem::path & filename );

    void input ( vector<double> input );
    void calculate ();

    // Gets strongest node
    int getAnswer ();

    // Gets if the network correctly guessed
    bool isAnswerCorrect ( vector<double> expectedAnswers );

    // Gets the output layer
    vector<double> getAnswerVector ();

    void SetParameters ( Parameters parameters );

    Parameters GetParameters() const;

    Parameters EmptyParameters ();
};