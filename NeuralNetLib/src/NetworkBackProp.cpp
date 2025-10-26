#include "NetworkBackProp.h"

using std::vector, std::tuple;

// Avoid initialised the interfaces twice
NetworkBackProp::NetworkBackProp(const Parameters& parameters, const Interface interface_) : Network(parameters, interface_, false) {

    if (interface_ == Interface::GPU) {
        // gpuInterface = std::make_unique<NetworkBackPropGPU>(this->parameters);
        // backPropGPUinterface = dynamic_cast<NetworkBackPropGPU*>(gpuInterface.value().get());
    }
    else {
        cpuInterface = std::make_unique<NetworkBackPropCPU>(this->parameters);
        backPropCPUinterface = dynamic_cast<NetworkBackPropCPU*>(cpuInterface.value().get());
    }
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const Parameters& parameters, Interface interface_) {
    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(parameters, interface_));
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const std::vector<size_t>& Structure, Interface interface_) {
    Parameters parameters(Structure);
    parameters.initParameters();

    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(parameters, interface_));
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const std::filesystem::path& filename, Interface interface_) {
    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(getParameters(filename), interface_));
}

TestResult NetworkBackProp::TrainSet(const vector<Test>& testSet, double learningRate) {
    return backPropCPUinterface.value()->TrainSet(testSet, learningRate);
}

TestResult NetworkBackProp::TestSet(const vector<Test>& testSet) {
    return backPropCPUinterface.value()->TestSet(testSet);
}