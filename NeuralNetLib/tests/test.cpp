#include <gtest/gtest.h>

#include "NeuralNet.h"

TEST(NeuralNetTests, Calculate) {
	std::vector<size_t> structure = { 3, 4, 2 };
	Network net(structure);

	std::vector<double> input = { 1.0, 1.0, 1.0 };
	net.input(input);

	EXPECT_NO_THROW(net.calculateGPU());
	std::vector<double> GPUoutput = net.getAnswerVector();
	net.calculateCPU();
	std::vector<double> CPUoutput = net.getAnswerVector();

	ASSERT_EQ(GPUoutput, CPUoutput);
}