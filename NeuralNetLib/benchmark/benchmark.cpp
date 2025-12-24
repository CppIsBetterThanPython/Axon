#include <benchmark/benchmark.h>
#include "Network.hpp"

constexpr size_t seed = 100;

static void BM_CalculateGPU(benchmark::State& state) {
	static size_t now = std::chrono::system_clock::now().time_since_epoch().count();
	static std::mt19937 randomEngine(seed);

	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };
	std::unique_ptr<Network> net = Network::createNetwork(structure, Network::Interface::GPU);

	std::vector<std::vector<double>> input = {};
	input.reserve(1000);
	for (size_t i = 0; i < 1000; i++) {
		input.push_back(std::vector<double>());
		input[i].reserve(structure[0]);

		for (int j = 0; j < structure[0]; j++) {
			input[i].push_back(RandomReal(randomEngine));
		}
	}

	for (auto _ : state) {
		state.PauseTiming();
		std::vector<std::vector<double>> input = {};
		input.reserve(1000);
		for (size_t i = 0; i < 1000; i++) {
			input.push_back(std::vector<double>());
			input[i].reserve(structure[0]);

			for (int j = 0; j < structure[0]; j++) {
				input[i].push_back(RandomReal(randomEngine));
			}
		}
		net->input(input);
		state.ResumeTiming();

		net->calculate();

		net->getAnswerVectors();
	}
}

static void BM_CalculateCPU(benchmark::State& state) {
	static size_t now = std::chrono::system_clock::now().time_since_epoch().count();
	static std::mt19937 randomEngine(seed);

	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };
	std::unique_ptr<Network> net = Network::createNetwork(structure, Network::Interface::CPU);

	std::vector<double> input;
	input.reserve(structure[0]);
	for (int i = 0; i < structure[0]; i++) {
		input.push_back(RandomReal(randomEngine));
	}

	for (auto _ : state) {
		for (size_t i = 0; i < 1000; i++) {
			state.PauseTiming();
			for (int i = 0; i < structure[0]; i++) {
				input[i] = RandomReal(randomEngine);
			}
			net->input(input);
			state.ResumeTiming();

			net->calculate();

			net->getAnswerVector();
		}
	}
}

BENCHMARK(BM_CalculateGPU);
BENCHMARK(BM_CalculateCPU);
BENCHMARK_MAIN();