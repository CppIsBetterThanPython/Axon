#include "NeuralNet.h"
#include "GPU.h"

NetworkGPU::NetworkGPU(Parameters& parameters) : NetworkBase(parameters.structure), parameters(parameters) {
	gpu = std::make_unique<GPU>();

	WeightBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
	BiasBuffers = std::vector<std::unique_ptr<cl::Buffer>>{};
	saveBuffers();
}

NetworkGPU::~NetworkGPU() = default;

// TODO: disallow input after calculate
void NetworkGPU::input(const std::vector<double>& input) {
	if (input.size() != inputLayerSize())
		throw std::invalid_argument("NetworkGPU::input: Input size does not match input layer size. Expected: " + std::to_string(inputLayerSize()) +
			" but got:" + std::to_string(input.size()));

	std::vector<double> extendedInput(largestLayer(), 0);

	std::copy(input.begin(), input.end(), extendedInput.begin());

	inputBuffer = std::make_unique<cl::Buffer>(
		gpu->context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * largestLayer(),
		extendedInput.data()
	);

	batchSize = 1;
}

void NetworkGPU::input(const std::vector<std::vector<double>>& input) {
	if (input.size() == 0)
		throw std::invalid_argument("NetworkGPU::input: Cannot have 0 batches in an input.");

	for (const std::vector<double>& batch : input)
		if (batch.size() != inputLayerSize())
			throw std::invalid_argument("NetworkGPU::input: Input size does not match input layer size. Expected: " + std::to_string(inputLayerSize()) +
				" but got:" + std::to_string(batch.size()));

	std::vector<double> extendedInput(largestLayer() * input.size(), 0);
	
	for (size_t i = 0; i < input.size(); i++)
		std::copy(input[i].begin(), input[i].end(), &extendedInput[inputLayerSize() * i]);

	inputBuffer = std::make_unique<cl::Buffer>(
		gpu->context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * extendedInput.size(),
		extendedInput.data()
	);

	batchSize = input.size();
}

/*
static void calculateLayer(
	GPU& gpu,
	const cl::Buffer& prevNodes,
	const cl::Buffer& weights,
	const cl::Buffer& biases,
	cl::Buffer& nodes,
	size_t prevNodesSize,
	size_t nodesSize) {

	const char* kernelSource = R"CLC(
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	__kernel void calculateLayer(
			__global double* prevNodes,
			__global double* weights,
			__global double* biases,
			__global double* nodes,
			ulong prevNodesSize) {
	
		int id = get_global_id(0);
		nodes[id] = 0.0;
		for (int i = 0; i < prevNodesSize; i++) {
			nodes[id] += weights[id * prevNodesSize + i] * prevNodes[i];
		}
		nodes[id] += biases[id];
	
		nodes[id] = 1.0 / (1.0 + exp(-nodes[id]));
	}
	)CLC";

	gpu.BuildKernel("calculateLayer", kernelSource);

	cl::Kernel& kernel = gpu.kernels["calculateLayer"];

	kernel.setArg(0, prevNodes);
	kernel.setArg(1, weights);
	kernel.setArg(2, biases);
	kernel.setArg(3, nodes);
	kernel.setArg(4, (cl_long)prevNodesSize);

	cl::NDRange global(nodesSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}


// TODO: Make largest layer calculations account for odd/even
void NetworkGPU::calculate() {
	// Make buffers ping pong
	GPU& gpu = *this->gpu;
	cl::Buffer bufferA(*inputBuffer);
	cl::Buffer bufferB(gpu.context, CL_MEM_READ_WRITE, sizeof(double) * largestPassLayer());

	cl::Buffer* prevLayerBuffer = &bufferA;
	cl::Buffer* curLayerBuffer = &bufferB;

	for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {
		size_t prevLayerSize = getStructure()[currentLayerIndex - 1];
		size_t curLayerSize = getStructure()[currentLayerIndex];

		cl::Buffer& weightsBuffer = *WeightBuffers[currentLayerIndex - 1];
		cl::Buffer& biasesBuffer = *BiasBuffers[currentLayerIndex - 1];

		calculateLayer(gpu, *prevLayerBuffer, weightsBuffer, biasesBuffer, *curLayerBuffer, prevLayerSize, curLayerSize);

		std::swap(curLayerBuffer, prevLayerBuffer);
	}

	outputBuffer = std::make_unique<cl::Buffer>(*prevLayerBuffer);
} */

static void calculateLayer(
	GPU& gpu,
	const cl::Buffer& prevNodes,
	const cl::Buffer& weights,
	const cl::Buffer& biases,
	cl::Buffer& nodes,
	size_t prevNodesSize,
	size_t nodesSize,
	size_t batchSize
) {

	const char* kernelSource = R"CLC(
	#pragma OPENCL EXTENSION cl_khr_fp64 : enablei
	__kernel void calculateLayer(
			__global double* prevNodes,
			__global double* weights,
			__global double* biases,
			__global double* nodes,
			ulong prevNodesSize,
			ulong curNodeSize) {
	
		int neuronID = get_global_id(0);
		int batchID = get_global_id(1);

		nodes[batchID * curNodeSize + neuronID] = 0.0;
		for (int i = 0; i < prevNodesSize; i++) {
			nodes[batchID * curNodeSize + neuronID] += weights[neuronID * prevNodesSize + i] * prevNodes[batchID * prevNodesSize + i];
		}
		nodes[batchID * curNodeSize + neuronID] += biases[neuronID];
		
		// Sigmoid function
		nodes[batchID * curNodeSize + neuronID] = 1.0 / (1.0 + exp(-nodes[batchID * curNodeSize + neuronID]));
	}
	)CLC";

	gpu.BuildKernel("calculateLayer", kernelSource);

	cl::Kernel& kernel = gpu.kernels["calculateLayer"];

	kernel.setArg(0, prevNodes);
	kernel.setArg(1, weights);
	kernel.setArg(2, biases);
	kernel.setArg(3, nodes);
	kernel.setArg(4, (cl_long)prevNodesSize);
	kernel.setArg(5, (cl_long)nodesSize);

	cl::NDRange global(nodesSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}

void NetworkGPU::calculate() {
	if (!inputBuffer)
		throw std::invalid_argument("NetworkGPU::calculate: Must have input.");

	// Make buffers ping pong
	cl::Buffer bufferA(*inputBuffer);
	cl::Buffer bufferB(gpu->context, CL_MEM_READ_WRITE, sizeof(double) * largestPassLayer() * batchSize);

	size_t prevBufferSize = largestLayer();
	size_t curBufferSize = largestPassLayer();

	// TODO: consider dropping the pointers and just swap the buffers
	cl::Buffer* prevLayerBuffer = &bufferA;
	cl::Buffer* curLayerBuffer = &bufferB;

	for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {
		const size_t prevLayerSize = getStructure(currentLayerIndex - 1);
		const size_t curLayerSize = getStructure(currentLayerIndex);

		cl::Buffer& weightsBuffer = *WeightBuffers[currentLayerIndex - 1];
		cl::Buffer& biasesBuffer = *BiasBuffers[currentLayerIndex - 1];

		calculateLayer(
			*gpu,
			*prevLayerBuffer,
			weightsBuffer,
			biasesBuffer,
			*curLayerBuffer,
			prevLayerSize,
			curLayerSize,
			batchSize
		);

		std::swap(curLayerBuffer, prevLayerBuffer);
		std::swap(curBufferSize, prevBufferSize);
	}

	outputBuffer = std::make_unique<cl::Buffer>(*prevLayerBuffer);
	// TODO: move this
	//gpu->queue.enqueueReadBuffer(*prevLayerBuffer, CL_TRUE, 0, sizeof(double) * (*this)[size() - 1].nodeData.size(), (*this)[size() - 1].nodeData.data());
}

std::vector<double> NetworkGPU::getAnswerVector() const {
	std::vector<double> answerVector(outputLayerSize());

	gpu->queue.enqueueReadBuffer(*outputBuffer, CL_TRUE, 0, sizeof(double) * outputLayerSize(), answerVector.data());

	return answerVector;
}

std::vector<std::vector<double>> NetworkGPU::getAnswerVectors() const {
	std::vector<double> answerVector(outputLayerSize() * batchSize);

	std::vector<std::vector<double>> answers;
	answers.reserve(batchSize);

	gpu->queue.enqueueReadBuffer(*outputBuffer, CL_TRUE, 0, sizeof(double) * outputLayerSize() * batchSize, answerVector.data());

	for (size_t i = 0; i < batchSize; i++) {
		answers.emplace_back(std::vector<double>(answerVector.begin() + outputLayerSize() * i, answerVector.begin() + outputLayerSize() * (i + 1)));
	}

	return answers;
}

void NetworkGPU::saveBuffers() {
	WeightBuffers.reserve(size());

	BiasBuffers.reserve(size());

	for (size_t i = 0; i < parameters.size; i++) {
		std::vector<double> layerWeights = flattenVector(parameters.weights[i]);
		std::vector<double> layerBiases(parameters.biases[i].begin(), parameters.biases[i].end());

		cl::Buffer weightBuffer(gpu->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerWeights.size(), layerWeights.data());
		cl::Buffer biasBuffer(gpu->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerBiases.size(), layerBiases.data());

		WeightBuffers.push_back(std::make_unique<cl::Buffer>(weightBuffer));
		BiasBuffers.push_back(std::make_unique<cl::Buffer>(biasBuffer));
	}
}