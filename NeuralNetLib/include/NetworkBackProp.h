#pragma once

#include "NeuralNet.h"

struct TestData {
    std::vector<double> values = {};
    explicit TestData(const std::vector<double>& values) : values(values) {}
    explicit TestData(std::vector<double>&& values) : values(std::move(values)) {}

    double& operator[](size_t i) { return values[i]; }
    const double operator[](size_t i) const { return values[i]; }

    const size_t size() const { return values.size(); }
};

struct Answer {
    std::vector<double> values = {};
    explicit Answer(const std::vector<double>& values) : values(values) {}
    explicit Answer(std::vector<double>&& values) : values(std::move(values)) {}

    double& operator[](size_t i) { return values[i]; }
    const double operator[](size_t i) const { return values[i]; }

    const size_t size() const { return values.size(); }
};

struct Test {
    TestData input;
    Answer expected;

    explicit Test(const TestData& data, const Answer& answer) : input(data), expected(answer) {}
    explicit Test(TestData&& data, Answer&& answer) : input(std::move(data)), expected(std::move(answer)) {}
};

struct TestResult {
    double cost;
    double accuracy;
};

class NetworkBackPropBase {

    virtual TestResult TestSet(const std::vector<Test>& testSet) = 0;
    virtual TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) = 0;
};

class NetworkBackPropCPU : public NetworkBackPropBase, public NetworkCPU {
private:
    using NetworkCPU::NetworkCPU;

    double getCost(std::vector<double> expectedAnswers);

    double getNodeGradient(const std::vector<double>& nextLayerNodeGradients, size_t nodePos, size_t currentLayerPos);
    double getWeightGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos, size_t weightPos);
    double getBiasGradient(double nextLayerNodeGradient, size_t layerPos, size_t nodePos);

    Parameters differentiate(std::vector<double> expectedAnswers);

    std::tuple< Parameters, double, bool> RunTrain(const Test& test);
    std::tuple< double, bool>             RunTest (const Test& test);
public:
    TestResult TestSet(const std::vector<Test>& testSet) override;
    TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;


private:
    //removes average gradient from the weights and biases
    void alterByGradient(const Parameters& averageGradient, double learningRate);
};

//class NetworkBackPropGPU : public NetworkBackPropBase, public NetworkGPU {
//private:
//    using NetworkGPU::NetworkGPU;
//
//public:
//    TestResult TestSet(const std::vector<Test>& testSet) override;
//    TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;
//};

class NetworkBackProp :
    public Network, public NetworkBackPropBase
{
    std::optional<NetworkBackPropCPU*> backPropCPUinterface;
    //std::optional<NetworkBackPropGPU*> backPropGPUinterface;

    NetworkBackProp(const Parameters& parameters, const Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
public:
    static std::unique_ptr<NetworkBackProp> createNetwork(const Parameters& parameters,          Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
    static std::unique_ptr<NetworkBackProp> createNetwork(const std::vector<size_t>& Structure,  Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
    static std::unique_ptr<NetworkBackProp> createNetwork(const std::filesystem::path& filename, Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);

    TestResult TestSet(const std::vector<Test>& testSet) override;
    TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;
};