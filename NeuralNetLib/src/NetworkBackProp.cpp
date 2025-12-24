#include "NetworkBackProp.hpp"

using std::vector, std::tuple;

// Avoid initialised the interfaces twice
NetworkBackProp::NetworkBackProp(const Parameters& parameters, const Interface interface_, const std::optional<size_t> seed) : Network(parameters, interface_, seed, false) {

    if (interface_ == Interface::GPU) {
        // gpuInterface = std::make_unique<NetworkBackPropGPU>(this->parameters);
        // backPropGPUinterface = dynamic_cast<NetworkBackPropGPU*>(gpuInterface.value().get());
    }
    else {
        cpuInterface = std::make_unique<NetworkBackPropCPU>(this->parameters);
        backPropCPUinterface = dynamic_cast<NetworkBackPropCPU*>(cpuInterface.value().get());
    }
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const Parameters& parameters, Interface interface_, std::optional<size_t> seed) {
    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(parameters, interface_, seed));
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const std::vector<size_t>& Structure, Interface interface_, std::optional<size_t> seed) {
    Parameters parameters(Structure);

    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(parameters, interface_, seed));
}

std::unique_ptr<NetworkBackProp> NetworkBackProp::createNetwork(const std::filesystem::path& filename, Interface interface_, std::optional<size_t> seed) {
    return std::unique_ptr<NetworkBackProp>(new NetworkBackProp(getParameters(filename), interface_, seed));
}

TestResult NetworkBackProp::TrainSet(const vector<Test>& testSet, double learningRate) {
    return backPropCPUinterface.value()->TrainSet(testSet, learningRate);
}

TestResult NetworkBackProp::TestSet(const vector<Test>& testSet) {
    return backPropCPUinterface.value()->TestSet(testSet);
}