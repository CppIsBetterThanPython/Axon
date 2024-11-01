#pragma once

#include <vector>
#include <tuple> // Include this for std::tuple
#include <utility>//for pair
#include <iostream>

using std::vector, std::tuple, std::string;

//to turn values into 0-1 range
inline double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

//to backpropogate
inline double SigmoidDerivative(double x) { return Sigmoid(x) * (1 - Sigmoid(x)); }

inline bool isInRange(double num, double lower, double upper) {
    return (num >= lower && num <= upper);
}

double RandomReal();

double XavierInitialization(int in, int out);

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

    double& operator[](int index) {
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

    Node& operator[](int index) {
        if (index < 0 || index >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return nodes[index];
    }
};


class Network {
    public:
    //layers[layer][node][weight]
    vector<Layer> layers;
    Layer* inputLayer;
    Layer* outputLayer;
    vector<size_t> structure;
    int networkSize;

    //constructor
    Network(vector<size_t> Structure);

    inline operator vector<Layer>() { return layers; }

    void resetWeights();

    void saveNetwork(string filename);

    void loadNetwork(string filename);

    //input vector to set input node layer
    void input(vector<double> input);

    //passes data through layers
    void calculate();

    //Gets strongest node
    int getAnswer();

    //Gets if the network correctly guessed
    bool isAnswerCorrect(vector<double> expectedAnswers);

    //Gets the output layer
    vector<double> getAnswerVector();

    double getCost(vector<double> expectedAnswers);

    vector<vector<vector<double>>> differentiate(vector<double> expectedAnswers);

private:
    //input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
    double getNodeGradient(vector<double> nextLayerNodeGradients, int nodePos, int currentLayerPos);

    //input the gradients of the next layers nodes, the index of the node you are getting the gradient of, and the index of the current layer
    double getGradient(double nextLayerNodeGradient, int layerPos, int nodePos, int weightPos);

    double getGradient(double nextLayerNodeGradient, int layerPos, int nodePos);

public:

    tuple< vector<vector<vector<double>>>, double, bool> Test(vector<vector<double>> testExample);

    tuple< vector<vector<vector<double>>>, double, double> TestSet(vector<vector<vector<double>>> testSet);
    
    //removes gradient from the weights and biases
    void alterByGradient(vector<vector<vector<double>>> averageGradient, double learningRate);

    //returns cost then accuracy
    vector<double> improveNetworkBackPropogation(vector<vector<vector<double>>> testSet, double learningRate);
};