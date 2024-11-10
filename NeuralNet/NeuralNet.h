#pragma once

#include <vector>
#include <tuple> // For returning multiple, different type values
#include <iostream>

using std::vector, std::tuple;

//to turn values into 0-1 range
inline double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

// Back propogation involves partial derivative, including that of the sigmoid function
inline double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

inline bool IsInRange(double num, double lower, double upper) {
    return (num >= lower && num <= upper);
}

// To set preliminary guesses for weights and biases
inline double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

// More efficient and stable initialisation
static inline double XavierInitialization(int in, int out) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal() * range;  // Scaled random value
    return x;
}

class Node {
public:
    size_t PrevLayerNodes;

    double data;
    double preSigmoidData;

    vector<double> Weights;
    double Bias;

    Node() : PrevLayerNodes(0), data(0.0), Weights({}), Bias(0.1), preSigmoidData(0.0) {}

    Node(int prevLayerNodes, int currentLayerNodes) : preSigmoidData(0.0), data(0.0), Bias(0.1) {
        PrevLayerNodes = prevLayerNodes;
        Weights = vector<double>(prevLayerNodes);
        for (int i = 0; i < PrevLayerNodes; i++) {
            Weights[i] = XavierInitialization(prevLayerNodes, currentLayerNodes);
        }
    }

    inline operator vector<double>() { return Weights; }

    inline double& operator[](int index) {
        if (index < 0 || index >= PrevLayerNodes) {
            throw std::out_of_range("Index out of bounds");
        }
        return Weights[index];
    }
};

class Layer {
public:
    size_t size;
    vector<Node> nodes;

    Layer() : size(0), nodes({}) {}

    Layer(int NodesAmount, int previousLayerNodes) {
        size = NodesAmount;
        nodes = vector<Node>(NodesAmount);
        for (int i = 0; i < NodesAmount; i++) {
            nodes[i] = Node(previousLayerNodes, NodesAmount);
        }
    }

    inline operator vector<Node>() { return nodes; }

    inline Node& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return nodes[index];
    }
};

class Network {
    friend class NNFile;
private:
    // layers[layer][node][weight]
    vector<Layer> layers;
    Layer* inputLayer;
    Layer* outputLayer;
public:
    int networkSize;
    vector<size_t> structure;

    Network(vector<size_t> Structure);

    inline operator vector<Layer>() { return layers; }

    inline Layer& operator[](int index) {
        if (index < 0 || index >= networkSize) {
            throw std::out_of_range("Index out of bounds");
        }
        return layers[index];
    }

    void resetWeights();

    void saveNetwork(std::string filename);
    void loadNetwork(std::string filename);

    void input(vector<double> input);
    void calculate();

    // Gets strongest node
    int getAnswer();

    // Gets if the network correctly guessed
    bool isAnswerCorrect(vector<double> expectedAnswers);

    // Gets the output layer
    vector<double> getAnswerVector();

    double getCost(vector<double> expectedAnswers);

    vector<vector<vector<double>>> differentiate(vector<double> expectedAnswers);

private:
    double getNodeGradient(vector<double> nextLayerNodeGradients, int nodePos, int currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, int layerPos, int nodePos, int weightPos);
    double getBiasGradient(double nextLayerNodeGradient, int layerPos, int nodePos);

public:
    tuple< vector<vector<vector<double>>>, double, bool> Test(vector<vector<double>> testExample);
    tuple< vector<vector<vector<double>>>, double, double> TestSet(vector<vector<vector<double>>> testSet);
    
    //removes average gradient from the weights and biases
    void alterByGradient(vector<vector<vector<double>>> averageGradient, double learningRate);

    //returns cost then accuracy
    vector<double> improveNetworkBackPropogation(vector<vector<vector<double>>> testSet, double learningRate);
};