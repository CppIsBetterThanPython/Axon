#include <gtest/gtest.h>

#include "NetworkBackProp.h"

inline bool IsInRange(double num, double lower, double upper) {
	return (num >= lower && num <= upper);
}

static std::vector<Test> generateTestSet(int size) {
	static size_t now = std::chrono::system_clock::now().time_since_epoch().count();
	static std::mt19937 randomEngine(now);

	std::vector<Test> testSet;

	testSet.reserve(size);

	for (int i = 0; i < size; i++) {
		std::vector<double> x = { static_cast<double>(rand() % 100) + RandomReal(randomEngine), static_cast<double>(rand() % 100) + RandomReal(randomEngine) };
		std::vector<double> y = { static_cast<double>(rand() % 100) + RandomReal(randomEngine), static_cast<double>(rand() % 100) + RandomReal(randomEngine) };

		std::sort(x.begin(), x.end());
		std::sort(y.begin(), y.end());

		std::vector<double> point = { static_cast<double>(rand() % 100) + RandomReal(randomEngine), static_cast<double>(rand() % 100) + RandomReal(randomEngine) };

		TestData input(std::vector<double>{ x[0], y[0], x[1], y[1], point[0], point[1] });

		Answer answer(std::vector<double>(2, 0));

		if (IsInRange(point[0], x[0], x[1]) && IsInRange(point[1], y[0], y[1])) {
			answer[0] = 1;
		}
		else {
			answer[1] = 1;
		}

		Test TestData = Test{ input, answer };

		testSet.push_back(TestData);
	}

	return testSet;
}

TEST(FileIOTests, basicIO) {
	static size_t now = std::chrono::system_clock::now().time_since_epoch().count();
	static std::mt19937 randomEngine(now);

	std::vector<size_t> structure = { 3, 4, 2 };

	const std::filesystem::path path = "tmp.nn";

	Parameters parameters(structure);
	parameters.initParameters(randomEngine);

	saveParameters(parameters, path);
	
	Parameters readParameters = getParameters(path);

	ASSERT_TRUE(readParameters == parameters);

	std::filesystem::remove(path);
}

TEST(ForwardPassTests, SingleInput) {
	std::vector<size_t> structure = { 3, 4, 2 };
	std::unique_ptr<Network> net = Network::createNetwork(structure, Network::Interface::GPU);

	std::vector<double> input = { 1.0, 1.0, 1.0 };

	net->input(input);
	EXPECT_NO_THROW(net->calculate());
	std::vector<double> GPUoutput = net->getAnswerVector();

	net->switchInterface();

	net->input(input);
	net->calculate();
	std::vector<double> CPUoutput = net->getAnswerVector();

	ASSERT_EQ(GPUoutput, CPUoutput);
}

TEST(ForwardPassTests, BatchedInput) {
	std::vector<size_t> structure = { 3, 4, 2 };
	std::unique_ptr<Network> net = Network::createNetwork(structure, Network::Interface::GPU);

	std::vector<std::vector<double>> inputs = {
		{1.0, 1.0, 1.0},
		{0.5, 0.5, 0.5},
		{0.29, 0.56, 0.78}
	};

	net->input(inputs);

	EXPECT_NO_THROW(net->calculate());
	std::vector<std::vector<double>> GPUoutputs = net->getAnswerVectors();

	net->switchInterface();

	net->input(inputs);
	net->calculate();
	std::vector<std::vector<double>> CPUoutputs = net->getAnswerVectors();

	ASSERT_EQ(GPUoutputs, CPUoutputs);
}

/*
TEST(BackwardsPassTests, Basic) {

	std::vector<size_t> structure = { 6, 2 };
	std::unique_ptr<NetworkBackProp> net = NetworkBackProp::createNetwork(structure);

	EXPECT_NO_THROW(net->TrainSet(generateTestSet(5), 0.01));
}
*/

TEST(RandomTests, Determisism) {
	std::vector<size_t> structure = { 3, 4, 2 };
	std::unique_ptr<Network> net1 = Network::createNetwork(structure, Network::Interface::GPU);
	std::unique_ptr<Network> net2 = Network::createNetwork(structure, Network::Interface::GPU);

	ASSERT_TRUE(net1->getNetworkParameters() == net2->getNetworkParameters());
}

TEST(RandomTests, RandomSeed) {
	std::vector<size_t> structure = { 3, 4, 2 };
	const size_t seed1 = 1;
	const size_t seed2 = 2;
	std::unique_ptr<Network> net1 = Network::createNetwork(structure, Network::Interface::GPU, seed1);
	std::unique_ptr<Network> net2 = Network::createNetwork(structure, Network::Interface::GPU, seed2);

	ASSERT_FALSE(net1->getNetworkParameters() == net2->getNetworkParameters());
}