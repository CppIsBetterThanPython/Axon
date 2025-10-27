#include <cstdlib> // For random numbers
#include <ctime>   // So is this

#include "NeuralNet.h"
#include "GPU.h"

using std::vector, std::tuple;

Network::Network(const Parameters& parameters, Interface interface_, bool initParameters)
    : NetworkBase(parameters.structure), parameters(parameters) {

    state = State::Ready;
    inputType = InputType::Singular;

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

// TODO: consider just adding to the batched queue if already inputted
// TODO: return error codes
// Input vector to set input node layer
void Network::input(const vector<double>& input) {

    if (state == State::Inputted) {
        std::cerr << "Network::input - Already inputted, use batched inputs for multiple inputs.";
        return;
    }
    else if (state == State::Calculated) {
        std::cerr << "Network::input - Already calculated, please retrieve output first.";
        return;
    }

    switch (interface_) {
    case (Interface::GPU):
        gpuInterface.value()->input(input);
        break;
    case (Interface::CPU):
        cpuInterface.value()->input(input);
        break;
    }

    state = State::Inputted;
    inputType = InputType::Singular;
}

void Network::input(const std::vector<std::vector<double>>& input) {

    if (state == State::Inputted) {
        std::cerr << "Network::input - Already inputted, use batched inputs for multiple inputs.";
        return;
    }
    else if (state == State::Calculated) {
        std::cerr << "Network::input - Already calculated, please retrieve output first.";
        return;
    }

    switch (interface_) {
    case (Interface::GPU):
        gpuInterface.value()->input(input);
        break;
    case (Interface::CPU):
        cpuInterface.value()->input(input);
        break;
    }

    state = State::Inputted;
    inputType = InputType::Batched;
}

// TODO: Do a quick calculation to check if CPU or GPU will be faster
void Network::calculate() {

    if (state == State::Calculated) {
        std::cerr << "Network::calculate - Already calculated.";
        return;
    }
    else if (state == State::Ready) {
        std::cerr << "Network::calculate - No input, please give an input first.";
        return;
    }

    switch (interface_) {
    case (Interface::GPU):
        gpuInterface.value()->calculate();
        break;
    case (Interface::CPU):
        cpuInterface.value()->calculate();
        break;
    }

    state = State::Calculated;
}

// Gets the output layer
std::vector<double> Network::getAnswerVector() {

    if (state == State::Ready) {
        std::cerr << "Network::getAnswerVector - Please input and calculate first.";
        return {};
    }
    else if (state == State::Inputted) {
        std::cerr << "Network::getAnswerVector - Please calculate first.";
        return {};
    }

    if (inputType == InputType::Batched) {
        std::cerr << "Network::getAnswerVector - Incorrect input type, please use \"Network::getAnswerVectors\" instead for retrieving batched inputs";
        return {};
    }

    std::vector<double> output;

    switch (interface_) {
    case (Interface::GPU):
        output = gpuInterface.value()->getAnswerVector();
        break;
    case (Interface::CPU):
        output = cpuInterface.value()->getAnswerVector();
        break;
    }

    state = State::Ready;

    return output;
}

std::vector<std::vector<double>> Network::getAnswerVectors() {

    if (state == State::Ready) {
        std::cerr << "Network::getAnswerVector - Please input and calculate first.";
        return {};
    }
    else if (state == State::Inputted) {
        std::cerr << "Network::getAnswerVector - Please calculate first.";
        return {};
    }

    if (inputType == InputType::Singular) {
        std::cerr << "Network::getAnswerVector - Incorrect input type, please use \"Network::getAnswerVector\" instead for retrieving singular inputs";
        return {};
    }

    std::vector<std::vector<double>> output;

    switch (interface_) {
    case (Interface::GPU):
        output = gpuInterface.value()->getAnswerVectors();
        break;
    case (Interface::CPU):
        output = cpuInterface.value()->getAnswerVectors();
        break;
    }

    state = State::Ready;

    return output;
}