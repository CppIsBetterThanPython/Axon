#include <cstdlib> // For random numbers
#include <ctime>   // So is this

#include "NeuralNet.h"
#include "GPU.h"

using std::vector, std::tuple;

Network::Network(const Parameters& parameters, Interface interface_, bool initParameters)
    : NetworkBase(parameters.structure), parameters(parameters) {

    this->interface_ = interface_;
    if (initParameters) {
        if (interface_ == Interface::GPU) {
            gpuInterface = std::make_unique<NetworkGPU>(this->parameters);
        }
        else {
            cpuInterface = std::make_unique<NetworkCPU>(this->parameters);
        }
    }
}

std::unique_ptr<Network> Network::createNetwork(const Parameters& parameters, Interface interface_) {
    return std::unique_ptr<Network>(new Network(parameters, interface_));
}

std::unique_ptr<Network> Network::createNetwork(const std::vector<size_t>& Structure, Interface interface_) {
    Parameters parameters(Structure);
    parameters.initParameters();

    return std::unique_ptr<Network>(new Network(parameters, interface_));
}

std::unique_ptr<Network> Network::createNetwork(const std::filesystem::path& filename, Interface interface_) {
    return std::unique_ptr<Network>(new Network(getParameters(filename), interface_));
}

// TODO: Better handle back ups
Network::~Network() {};


// TODO: Make this a set, not a switch
void Network::switchInterface() {

    if (interface_ == Interface::GPU) {
        cpuInterface = std::make_unique<NetworkCPU>(this->parameters);
        gpuInterface.reset();

        interface_ = Interface::CPU;
    }
    else {
        gpuInterface = std::make_unique<NetworkGPU>(this->parameters);
        cpuInterface.reset();

        interface_ = Interface::GPU;
    }
}

// Input vector to set input node layer
void Network::input(const vector<double>& input) {

    if (interface_ == Interface::GPU) {
        if (gpuInterface) {
            gpuInterface.value()->input(input);
            return;
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    else {
        cpuInterface.value()->input(input);
    }
}

// TODO: Do a quick calculation to check if CPU or GPU will be faster
// Passes data through layers
void Network::calculate() {
    if (interface_ == Interface::GPU) {
        if (gpuInterface) {
            gpuInterface.value()->calculate();
            return;
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    cpuInterface.value()->calculate();
}

// Gets the output layer
vector<double> Network::getAnswerVector() const {
    if (interface_ == Interface::GPU) {
        if (gpuInterface) {
            return gpuInterface.value()->getAnswerVector();
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    return cpuInterface.value()->getAnswerVector();
}

void Network::input(const std::vector<std::vector<double>>& input) {
    if (interface_ == Interface::GPU) {
        if (gpuInterface) {
            return gpuInterface.value()->input(input);
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    return cpuInterface.value()->input(input);
}

std::vector<std::vector<double>> Network::getAnswerVectors() const {
    if (interface_ == Interface::GPU) {
        if (gpuInterface) {
            return gpuInterface.value()->getAnswerVectors();
        }
        std::cerr << "Had to fall Back to CPU." << std::endl;
    }
    return cpuInterface.value()->getAnswerVectors();
}