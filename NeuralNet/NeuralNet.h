#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>
#include <tuple> // Include this for std::tuple
#include <utility>//for pair

using std::vector;
using std::tuple;

double Sigmoid(double x);

double SigmoidDerivative(double x);

bool isInRange(double num, double lower, double upper);

double RandomReal();

class Node {
public:
    int PrevLayerNodes;

    double data;
    double preSigmoidData;

    vector<double> Weights;
    double Bias;

    Node() : PrevLayerNodes(0), data(0.0), Weights({}), Bias(0.0), preSigmoidData(0.0) {}

    Node(int prevLayerNodes) : preSigmoidData(0.0), data(0.0) {
        PrevLayerNodes = prevLayerNodes;
        Weights = vector<double>(prevLayerNodes);
        for (int i = 0; i < PrevLayerNodes; i++) {
            Weights[i] = RandomReal();
        }
        Bias = RandomReal();
    }

    double& operator[](int index) {
        if (index < 0 || index >= PrevLayerNodes) {
            throw std::out_of_range("Index out of bounds");
        }
        return Weights[index];
    }
};

class Layer {
public:
    int size;
    vector<Node> nodes;

    Layer() : size(0), nodes({}) {}

    Layer(int NodesAmount, int previousLayerNodes) {
        size = NodesAmount;
        nodes = vector<Node>(NodesAmount);
        for (int i = 0; i < NodesAmount; i++) {
            nodes[i] = Node(previousLayerNodes);
        }
    }

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
    vector<int> structure;
    int networkSize;

    //constructor
    Network(vector<int> Structure);

    void resetWeights();

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

#endif