#include <benchmark/benchmark.h>
#include "NeuralNet.h"

static void BM_CalculateGPU(benchmark::State& state) {
	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };
	Network net(structure);

	std::vector<double> input;
	input.reserve(structure[0]);
	for (int i = 0; i < structure[0]; i++) {
		input.push_back(RandomReal());
	}
	net.input(input);
	net.calculateGPU();

	for (auto _ : state) {
		state.PauseTiming();
		for (int i = 0; i < structure[0]; i++) {
			input[i] = RandomReal();
		}
		net.input(input);
		state.ResumeTiming();

		net.calculateGPU();
	}
}

static void BM_CalculateCPU(benchmark::State& state) {
	std::vector<size_t> structure = { 1000, 1000, 1000, 1000, 1000 };
	Network net(structure);

	std::vector<double> input;
	input.reserve(structure[0]);
	for (int i = 0; i < structure[0]; i++) {
		input.push_back(RandomReal());
	}
	net.input(input);

	for (auto _ : state) {
		state.PauseTiming();
		for (int i = 0; i < structure[0]; i++) {
			input[i] = RandomReal();
		}
		net.input(input);
		state.ResumeTiming();

		net.calculateCPU();
	}
}

BENCHMARK(BM_CalculateGPU);
BENCHMARK(BM_CalculateCPU);
BENCHMARK_MAIN();