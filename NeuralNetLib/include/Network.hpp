#pragma once

#include <filesystem>
#include <optional>

#include "pch.h"

#include "Parameters.hpp"
#include "NetworkBase.hpp"

// Forward declaration of GPU interface class that is managed internally.
// This is not meant to be used by itself.
class NetworkGPU;

class NetworkCPU;

// TODO: Add a optimising option, which allows for the class to decide when to use which interface.
class Network : public NetworkBase {
public:
    enum class Interface { CPU, GPU };
    enum class State { Ready, Inputted, Calculated };
    enum class InputType { Singular, Batched };
protected:
    std::mt19937 randomEngine;
    size_t seed;

    Parameters parameters;

    mutable Interface interface_;
    mutable State state;
    mutable InputType inputType;

    // Unique pointer is for polymorphism for other network types.
    std::optional<std::unique_ptr<NetworkCPU>> cpuInterface;
    std::optional<std::unique_ptr<NetworkGPU>> gpuInterface;

    // initParameters is to defer initialisation to derived classes to avoid initialising twice.
    Network(const Parameters& parameters, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed, bool initParameters = true);

public:
    static std::unique_ptr<Network> createNetwork(const Parameters& parameters, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);
    static std::unique_ptr<Network> createNetwork(const std::vector<size_t>& Structure, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);
    static std::unique_ptr<Network> createNetwork(const std::filesystem::path& filename, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);

    virtual void switchInterface();

    ~Network ();

    void input(const std::vector<std::vector<double>>&);
    void input (const std::vector<double>& input ) override;
    void calculate () override;
    [[nodiscard]]std::vector<double> getAnswerVector() const override;
    [[nodiscard]]std::vector<std::vector<double>> getAnswerVectors() const override;

    virtual bool saveNetwork(const std::filesystem::path& filename) const;
    virtual bool loadNetwork(const std::filesystem::path& filename);

    size_t getSeed() const { return seed; }

    // This is for testing mainly
    const Parameters& getNetworkParameters() { return parameters; }
};

Parameters getParameters(const std::filesystem::path& filePath);
bool saveParameters(const Parameters& parameters, const std::filesystem::path& filePath);