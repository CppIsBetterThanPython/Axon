#pragma once

#include <vector>
#include <iostream>

using std::vector, std::tuple;

// To turn values into 0-1 range
static inline double Sigmoid(double x) { return 1 / (1 + exp(-x)); }

// For random number to use in Xavier Initialization
inline double RandomReal() { return (static_cast<double>(rand()) / RAND_MAX) * 2 - 1; }

// More efficient and stable initialisation
static inline double XavierInitialization(int in, int out) {
    double range = sqrt(6.0 / (in + out));
    double x = RandomReal() * range;  // Scaled random value
    return x;
}

class Network {
public:
    friend class NNFile;
protected:

    class Layer {
    public:
        friend class NNFile;
    protected:

        class Node {
        public:
            double data            = 0.0,
                   Bias            = 0.1,
                   preSigmoidData  = 0.0;

            vector<double> Weights = {};

            Node() : data(0.0), Weights({}), Bias(0.1), preSigmoidData(0.0) {}

            Node(int prevLayerNodes, int currentLayerNodes) : preSigmoidData(0.0), data(0.0), Bias(0.1) {
                Weights = vector<double>(prevLayerNodes);
                for (int i = 0; i < prevLayerNodes; i++) {
                    Weights[i] = XavierInitialization(prevLayerNodes, currentLayerNodes);
                }
            }

            inline operator std::vector<double>() const {
                std::vector<double> controlables = Weights;  // Copy Weights into a new vector
                controlables.push_back(Bias);  // Add Bias to the vector
                return controlables;  // Return the result
            }

            inline operator double() { return data; }

            inline double& operator[](size_t index) {
                if (index >= Weights.size())
                    throw std::out_of_range("Index out of bounds");

                return Weights[index];
            }
        };

    public:
        size_t       size  = 0;
        vector<Node> nodes = {};

        Layer() : size(0), nodes({}) {}

        Layer(int NodesAmount, int previousLayerNodes) {
            size = NodesAmount;
            nodes = vector<Node>(NodesAmount);
            for (int i = 0; i < NodesAmount; i++)
                nodes[i] = Node(previousLayerNodes, NodesAmount);
        }

        inline operator vector<Node>() { return nodes; }

        inline operator vector<double>& () {
            vector<double> dataVect = {};
            for (double data : nodes)
                dataVect.push_back(data);

            return dataVect;
        }

        inline Node& operator[](size_t index) {
            if (index >= size)
                throw std::out_of_range("Index out of bounds");

            return nodes[index];
        }
    };

public:

    // Structure to store basic parameters of the n
    struct Parameters {
        vector<vector<vector<double>>> data = {};
        size_t size = 0;
        vector<size_t> structure = {};
    };

protected:
    // layers[layer][node][weight]
    vector<Layer> layers;
    Layer* inputLayer;
    Layer* outputLayer;
public:
    size_t networkSize;
    vector<size_t> structure;

    Network (vector<size_t> Structure);

    inline operator vector<Layer>() { return layers; }

    inline Layer& operator[](size_t index) {
        if (index >= networkSize)
            throw std::out_of_range("Index out of bounds");
        
        return layers[index];
    }

    void resetWeights ();

    void saveNetwork ( std::string filename );
    void loadNetwork ( std::string filename );

    void input ( vector<double> input );
    void calculate ();

    // Gets strongest node
    int getAnswer ();

    // Gets if the network correctly guessed
    bool isAnswerCorrect ( vector<double> expectedAnswers );

    // Gets the output layer
    vector<double> getAnswerVector();

    void SetParameters(Parameters parameters);

    Parameters GetParameters();

    Parameters EmptyParameters();
};