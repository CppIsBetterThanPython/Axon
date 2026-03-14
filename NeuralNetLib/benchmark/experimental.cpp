#include <benchmark/benchmark.h>
#include "Network.hpp"
#include "../src/detail/NetworkCPU.hpp"

constexpr size_t seed = 100;

static void BM_CalculateCPU(benchmark::State& state) {
	static std::mt19937 randomEngine(seed);

	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };

	axon::Parameters parameters(structure);

	parameters.initParameters(randomEngine);

	axon::NetworkCPU networkCPU(parameters);

	std::vector<double> input(1000);

	for (auto _ : state) {
		state.PauseTiming();
		for (int i = 0; i < structure[0]; i++) {
			input[i] = RandomReal(randomEngine);
		}
		networkCPU.input(input);
		state.ResumeTiming();

		networkCPU.calculate();
	}
}

static void BM_CalculateCPUB(benchmark::State& state) {
	static std::mt19937 randomEngine(seed);

	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };

	axon::Parameters parameters(structure);

	parameters.initParameters(randomEngine);

	axon::NetworkCPU networkCPU(parameters);

	std::vector<double> input(1000);

	for (auto _ : state) {
		state.PauseTiming();
		for (int i = 0; i < structure[0]; i++) {
			input[i] = RandomReal(randomEngine);
		}
		networkCPU.input(input);
		state.ResumeTiming();

		networkCPU.calculateB();
	}
}

static void BM_CalculateCPUC(benchmark::State& state) {
	static std::mt19937 randomEngine(seed);

	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };

	axon::Parameters parameters(structure);

	parameters.initParameters(randomEngine);

	axon::NetworkCPU networkCPU(parameters);

	std::vector<double> input(1000);

	for (auto _ : state) {
		state.PauseTiming();
		for (int i = 0; i < structure[0]; i++) {
			input[i] = RandomReal(randomEngine);
		}
		networkCPU.input(input);
		state.ResumeTiming();

		networkCPU.calculatePassC();
	}
}

BENCHMARK(BM_CalculateCPU);
BENCHMARK(BM_CalculateCPUB);
BENCHMARK(BM_CalculateCPUC);
BENCHMARK_MAIN();