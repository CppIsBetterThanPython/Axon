#pragma once

#include "Network.hpp"

#include "NetworkBackPropBase.hpp"

// TODO: REMOVE THIS
#include "NetworkBackPropCPU.hpp"

//class NetworkBackPropGPU;
//class NetworkBackPropCPU;

namespace axon {

    class NetworkBackProp :
        public Network, public NetworkBackPropBase
    {
        //std::optional<std::weak_ptr<NetworkBackPropCPU>> backPropCPUinterface;
        // TODO: MAKE THIS A WEAK PTR
        std::optional<NetworkBackPropCPU*> backPropCPUinterface;
        //std::optional<NetworkBackPropGPU*> backPropGPUinterface;

        NetworkBackProp(const Parameters& parameters, const Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
    public:
        static std::unique_ptr<NetworkBackProp> createNetwork(const Parameters& parameters, Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
        static std::unique_ptr<NetworkBackProp> createNetwork(const std::vector<size_t>& Structure, Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);
        static std::unique_ptr<NetworkBackProp> createNetwork(const std::filesystem::path& filename, Interface interface_ = Interface::CPU, std::optional<size_t> seed = defaultSeed);

        void switchInterface() override;

        TestResult TestSet(const std::vector<Test>& testSet) override;
        TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) override;
    };

}