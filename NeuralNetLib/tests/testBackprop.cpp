#include <gtest/gtest.h>

#define private public
#include "NetworkBackProp.hpp"
#undef private

TEST(BackpropagationGPU, LoadBuffers) {
	std::vector<size_t> structure = { 6, 8, 2 };
	std::unique_ptr<axon::NetworkBackProp> net = axon::NetworkBackProp::createNetwork(structure, axon::Network::Interface::GPU);

	net.

	
}
