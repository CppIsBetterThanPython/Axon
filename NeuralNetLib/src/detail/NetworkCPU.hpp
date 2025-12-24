#pragma once

#include "pch.h"

#include "Parameters.hpp"
#include "NetworkBase.hpp"

// Interface for networks with CPU, never exists on its own
class NetworkCPU : public NetworkBase {
protected:
    // Reference to paramaters stored in Network hyperclass, NetworkCPU is contained within Network, the reference is never invalidated
    Parameters& parameters;
    std::vector<double> nodeDataRaw;
    std::vector<std::span<double>> nodeData;

    std::vector<std::vector<double>> batchedInputs;
    std::vector<std::vector<double>> batchedOutputs;
public:

    NetworkCPU(Parameters& parameters);

    void input(const std::vector<std::vector<double>>&) override;
    void input(const std::vector<double>&) override;

    void calculate() override;
private:
    void calculatePass();
public:

    [[nodiscard]] std::vector<double> getAnswerVector() const override;
    [[nodiscard]] std::vector<std::vector<double>> getAnswerVectors() const override;
};