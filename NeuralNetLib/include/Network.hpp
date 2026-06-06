#pragma once

#include <filesystem>
#include <optional>

#include "pch.h"

#include "Parameters.hpp"
#include "NetworkBase.hpp"
#include "NetworkError.hpp"

namespace axon {

    // Forward declaration of GPU interface class that is managed internally.
    // This is not meant to be used by itself.
    class NetworkGPU;

    // Forward declaration of CPU interface class that is managed internally.
    // This is not meant to be used by itself.
    class NetworkCPU;

    // TODO: Add an optimising option, which allows for the class to decide when to use which interface.
    // TODO: Maybe force the Network to only own one interface, but in that case i would have to somehow cache other interfaces when testing which is fastest at the start
    class Network : public NetworkBase {
    public:
        enum class Interface { CPU, GPU };
        enum class State { Ready, Inputted, Calculated };
        enum class InputType { Singular, Batched };

        enum class Function : uint8_t { Sigmoid };
        enum class Initialisation : uint8_t { Xavier };
    protected:
        std::mt19937 randomEngine;
        // Possibly make this const
        size_t seed;

        Parameters parameters;

        mutable Interface interface_;
        mutable State state;
        mutable InputType inputType;

    public:
        Function activationFunction;
        Function outputFunction;

        const Initialisation weightInitialisation;
        const Initialisation biasInitialisation;
    protected:

        // Unique pointer is for polymorphism for other network types and to allow instanciation without the full type definition.
        std::optional<std::unique_ptr<NetworkCPU>> cpuInterface;
        std::optional<std::unique_ptr<NetworkGPU>> gpuInterface;

        // initParameters is to defer initialisation to derived classes to avoid initialising twice.
        Network(const Parameters& parameters, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed, bool initInterface = true);

    public:
        static std::unique_ptr<Network> createNetwork(const Parameters& parameters, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);
        static std::unique_ptr<Network> createNetwork(const std::vector<size_t>& Structure, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);
        //static std::unique_ptr<Network> createNetwork(const std::filesystem::path& filename, Interface interface_ = Interface::GPU, std::optional<size_t> seed = defaultSeed);

        ~Network();

	/**
	 * @brief Switches the current compute interface
	 *
	 * @details Tries to swap the current interface to the opposite. May fail if no OpenCL instance can be made.
	 */
        virtual void switchInterface();

	/**
	 * @brief Inputs values into the network and changes internal state to reflect this.
	 *
	 * @details This only stores the values, it does not compute anything. If you call it again before calculating it will fail and keep the previous input.
	 */
        void input(const std::vector<double>& input) override;
	/**
	 * @brief \copybrief input(const std::vector<double>&)
	 *
	 * @details This only stores the values, it does not compute anything. It you call it again before calculating it will fail and keep the previous input. This version takes multiple inputs to compute in parallel.
	 *
	 * @param 
	 */
        void input(const std::vector<std::vector<double>>& input);
	/**
	 * @brief Performs a forward pass.
	 *
	 * @details Passes data through the layers of the network. Must happen after input.
	 */
        void calculate() override;
	/**
	 * @brief Returns the activation layer of the network.
	 *
	 * @details Only works when there was only one input. Must happen after calculation.
	 */
        [[nodiscard]] std::vector<double> getAnswerVector() const override;
	/**
	 * @brief Returns the activation layer of the network for all inputs.
	 *
	 * @details Must happen after calculation.
	 */
        [[nodiscard]] std::vector<std::vector<double>> getAnswerVectors() const override;

    private:
        friend std::unique_ptr<Network> loadNetwork(std::error_code& ec, std::ifstream& file);
    public:

        size_t getSeed() const { return seed; }

        Function getActivationFunction() const { return activationFunction; }
        Function getOuputFunction() const      { return outputFunction; }

        // This is for testing mainly
        const Parameters& getNetworkParameters() const { return parameters; }
    };

    std::error_code saveNetwork(const Network& network, const std::filesystem::path& filePath);
    std::unique_ptr<Network> loadNetwork(const std::filesystem::path& filePath, std::error_code& ec);

    std::string getFileVersion();
}
