#include "GPU.hpp"
#include <CL/cl.h>
#include "NetworkBackPropGPU.hpp"
#include "gpuUtils.hpp"

namespace axon {

NetworkBackPropGPU::NetworkBackPropGPU(Parameters& parameters) : NetworkGPU(parameters) {

	layerBuffers.resize(parameters.size);

}

void NetworkBackPropGPU::backPropCalculate(size_t batchSize) {

	for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {
		const size_t prevLayerSize = getStructure(currentLayerIndex - 1);
		const size_t curLayerSize = getStructure(currentLayerIndex);

		cl::Buffer& weightsBuffer = *WeightBuffers[currentLayerIndex - 1];
		cl::Buffer& biasesBuffer = *BiasBuffers[currentLayerIndex - 1];

		calculateLayer(
			*gpu,
			*layerBuffers[currentLayerIndex - 1],
			weightsBuffer,
			biasesBuffer,
			*layerBuffers[currentLayerIndex],
			prevLayerSize,
			curLayerSize,
			batchSize
		);
	}

}

static void getCosts(
	GPU& gpu,
	const cl::Buffer& predicted,
	const cl::Buffer& expected,
	cl::Buffer& costs,
	size_t outputLayerSize,
	size_t batchSize
) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double MSE(double predicted, double expected) {
	return pow(predicted - expected, 2);
}

__kernel void getCost(
		__global const double* predicted,	// Gradients that the parameters want to change by
		__global const double* expected,
		__global doube* costs,
		size_t outputLayerSize,
		size_t batchSize) {
	
	int neuronID = get_global_id(0);
	int batchID = get_global_id(1);

	costs[neuronID * batchSize + batchID] = MSE(predicted[outputLayerSize * batchID + neuronID], expected[outputLayerSize * batchID + neuronID]);
}
)CLC";

	gpu.BuildKernel("getCost", kernelSource);

	cl::Kernel& weightKernel = gpu.kernels["getCost"];

	weightKernel.setArg(0, predicted);
	weightKernel.setArg(1, expected);
	weightKernel.setArg(2, costs);
	weightKernel.setArg(3, outputLayerSize);
	weightKernel.setArg(4, batchSize);

	cl::NDRange global(outputLayerSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, global);
	gpu.queue.finish();
}

double getCost(
	GPU& gpu,
	const cl::Buffer& predicted,
	const cl::Buffer& expected,
	size_t outputLayerSize,
	size_t batchSize
) {
	cl::Buffer costs(
		gpu.context,
		CL_MEM_READ_WRITE,
		sizeof(double) * batchSize
	);

	getCosts(
		gpu,
		predicted,
		expected,
		costs,
		outputLayerSize,
		batchSize
	);

	return reduce(gpu, costs, batchSize);
}

/**
 * @brief Calculates the Deltas for the output layer
 *
 * This function computes the Deltas (d/dz Error) of the output layer of a fully connected neural network.
 * It does this using an OpenCL GPU compute kernel.
 * It allows for processing multiple backwards passes at once.
 *
 * @param gpu GPU execution context.
 * @param expectedValues Expected values for the output nodes.
 * @param outputNodes The values of the output nodes.
 * @param deltas The buffer of deltas that is written to.
 * @param nodesSize The amount of nodes in the output layer.
 * @param batchSize The amount of batches being proccessed.
 */
void outputLayerDeltas(
	GPU& gpu,
	const cl::Buffer& expectedValues,
	const cl::Buffer& outputNodes,
	cl::Buffer& deltas,
	size_t nodesSize,
	size_t batchSize
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static inline double MSEderivative(double predicted, double expected) {
	return 2 * (predicted - expected);
}

static inline double sigmoidDerivative(double sigmoid) {
	return sigmoid * (1 - sigmoid);
}

__kernel void outputLayerDeltas(
		__global const double* expectedValues,	// Expected values for the Output Nodes
		__global const double* outputNodes,	// Values of the Output Nodes
		__global double* deltas,		// Deltas for the Output Layer
		size_t nodeSize,
) {
	
	int deltaID = get_global_id(0);
	int batchID = get_global_id(1);

	// dE/da
	double nodeGradient =
		MSEderivative(
			outputNodes[batchID * nodeSize + deltaID],
			expectedValues[batchID * nodeSize + deltaID]
		);
	double node = outputNodes[batchID * nodeSize + deltaID];
	double nodeDerivative = sigmoidDerivative(node);

	deltas[batchID * nodeSize + deltaID] = nodeDerivative * nodeGradient;
}
)CLC";

	gpu.BuildKernel("outputLayerDeltas", kernelSource);

	cl::Kernel& kernel = gpu.kernels["outputLayerDeltas"];

	kernel.setArg(0, expectedValues);
	kernel.setArg(1, outputNodes);
	kernel.setArg(2, deltas);
	kernel.setArg(3, nodesSize);

	cl::NDRange global(nodesSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();

}

/**
 * @brief Calculates the Deltas for any (non-output) layer
 *
 * This function computes the Deltas (d/dz Error) of the given layer of a fully connected neural network.
 * It does this using an OpenCL GPU compute kernel.
 * It allows for processing multiple backwards passes at once.
 *
 * @param gpu GPU execution context.
 * @param weights Weights from current layer to previous layer.
 * @param nodes The values of the current layer nodes.
 * @param deltas The buffer of deltas that is written to.
 * @param prevLayerSize The amount of nodes in the previous layer.
 * @param layerSize The amount of nodes in the current layer.
 * @param batchSize The amount of batches being proccessed.
 */
void layerDeltas(
	GPU& gpu,
	const cl::Buffer& weights,
	const cl::Buffer& prevDeltas,
	const cl::Buffer& nodes,
	cl::Buffer& deltas,
	size_t prevLayerSize,
	size_t layerSize,
	size_t batchSize
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static inline double sigmoidDerivative(double sigmoid) {
	return sigmoid * (1 - sigmoid);
}

__kernel void layerDeltas(
		__global const double* weights,		// Weights from current layer to previous layer
		__global const double* prevDeltas,	// The deltas of the previous layer
		__global const double* nodes		// The values of the nodes of the current layer
		__global double* deltas,		// Deltas for the current layer
		size_t layerSize,
		size_t prevLayerSize
) {
	
	int deltaID = get_global_id(0);
	int batchID = get_global_id(1);

	// dE/da
	double nodeGradient = 0.0;

	for (size_t i = 0; i < prevLayerSize; i++) {
		nodeGradient += weights[i * prevLayerSize + deltaID] * prevDeltas[batchID * prevLayerSize + i];
	}

	double node = nodes[batchID * layerSize + deltaID];
	// dz/da
	double nodeDerivative = sigmoidDerivative(node);

	deltas[batchID * layerSize + deltaID] = nodeDerivative * nodeGradient;
}
)CLC";

	gpu.BuildKernel("layerDeltas", kernelSource);

	cl::Kernel& kernel = gpu.kernels["layerDeltas"];

	kernel.setArg(0, weights);
	kernel.setArg(1, prevDeltas);
	kernel.setArg(2, nodes);
	kernel.setArg(3, deltas);
	kernel.setArg(4, layerSize);
	kernel.setArg(5, prevLayerSize);

	cl::NDRange global(layerSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}


void weightGradients(
	GPU& gpu,
	const cl::Buffer& nodes,
	const cl::Buffer& prevDeltas,
	cl::Buffer& weightGradients,
	size_t layerSize,
	size_t prevLayerSize,
	size_t batchSize
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void weightGradients(
		__global const double* nodes,		// The values of the nodes of the current layer
		__global const double* prevDeltas,	// The deltas of the previous layer
		__global double* weightGradients,	// Weight gradients of the weights from the current layer to theprevious layer
		__local double* localData,		// Local array that stores the intermediate sums
		size_t layerSize,
		size_t prevLayerSize,
		size_t batchSize
) {
	
	size_t batchID = get_global_id(0);
	size_t localBatchID = get_local_id(0);
	size_t groupID = get_group_id(0);
	size_t localBatchSize = get_local_size(0);

	size_t localLayerSize = get_local_size(1);
	size_t localLayerID = get_local_id(1);
	size_t layerID = get_global_id(1);

	size_t localNodeID = get_local_id(2);
	size_t nodeID = get_global_id(2);

	__local double* localBatches = localData + localNodeID * localLayerSize * localBatchSize + localLayerID * localBatchSize;
	
	if (batchID < batchSize && layerID < layerSize && nodeID < prevLayerSize) {
		localBatches[localBatchID] =
			prevDeltas[batchID * prevLayerSize + nodeID] *
			nodes[batchID * layerSize + layerID];
	} else {
		localBatches[localBatchID] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (size_t stride = localBatchSize / 2; stride > 0; stride /= 2) {
		if (localBatchID < stride) {
			localBatches[localBatchID] += localBatches[localBatchID + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localBatchID == 0) {
		weightGradients[layerID * prevLayerSize * get_num_groups(0) + nodeID * get_num_groups(0) + groupID] = localBatches[0];
	}
}
)CLC";

	gpu.BuildKernel("weightGradients", kernelSource);

	cl::Kernel& kernel = gpu.kernels["weightGradients"];

	size_t localBatchSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t localLayerSize = std::min((size_t)256, std::min(gpu.maxWG / localBatchSize, (size_t)1));
	size_t localNodeSize = std::min((size_t)256, std::min(gpu.maxWG / localLayerSize, (size_t)1));

	size_t workGroupSize = localBatchSize * localLayerSize * localNodeSize;

	kernel.setArg(0, nodes);
	kernel.setArg(1, prevDeltas);
	kernel.setArg(2, weightGradients);
	kernel.setArg(3, cl::Local(sizeof(double) * workGroupSize));
	kernel.setArg(4, layerSize);
	kernel.setArg(5, prevLayerSize);
	kernel.setArg(6, batchSize);

	cl::NDRange global(batchSize, layerSize, prevLayerSize);
	cl::NDRange local(localBatchSize, localLayerSize, localNodeSize);
	
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	gpu.queue.finish();

	size_t strideSize = (batchSize - 1) / localBatchSize + 1;

	reduceStrided(
		gpu,
		weightGradients,
		strideSize,
		layerSize * prevLayerSize
	);
}


void modifyParameters(
	GPU& gpu,
	const cl::Buffer& weightGradients,
	const cl::Buffer& biasGradients,
	cl::Buffer& weights,
	cl::Buffer& biases,
	double learningRate,
	size_t nodesSize,
	size_t prevNodesSize // Prev relative to forward pass
) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void modifyParameters(
		__constant double* gradients,	// Gradients that the parameters want to change by
		__global double* parameters,
		double learningRate) {
	
	int elementID = get_global_id(0);

	parameters[elementID] -= learningRate * gradients[elementID];
}
)CLC";

	gpu.BuildKernel("modifyParameters", kernelSource);

	cl::Kernel& weightKernel = gpu.kernels["modifyParameters"];

	weightKernel.setArg(0, weightGradients);
	weightKernel.setArg(1, weights);
	weightKernel.setArg(2, (cl_double)learningRate);

	cl::NDRange weightRange(nodesSize * prevNodesSize);
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, weightRange);

	cl::Kernel& biasKernel = gpu.kernels["modifyParameters"];

	biasKernel.setArg(0, biasGradients);
	biasKernel.setArg(1, biases);
	biasKernel.setArg(2, (cl_double)learningRate);

	cl::NDRange biasRange(nodesSize);
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, biasRange);

	gpu.queue.finish();
}

inline void NetworkBackPropGPU::calculateGradients(
	const cl::Buffer& expectedBuffer,
	double learningRate,
	size_t batchSize,
	size_t prevBatchSize
) {

	// TODO: Consider double buffers so one can be read while the other is written.
	cl::Buffer weightGradientBuffer(
		gpu->context,
		CL_MEM_READ_WRITE,
		sizeof(double) * batchSize * maxWeightSize()
	);

	cl::Buffer prevDeltas(
		gpu->context,
		CL_MEM_READ_WRITE,
		sizeof(double) * batchSize * largestLayer()
	);

	cl::Buffer curDeltas(
		gpu->context,
		CL_MEM_READ_WRITE,
		sizeof(double) * batchSize * largestLayer()
	);

	outputLayerDeltas(
		*gpu,
		expectedBuffer,
		*layerBuffers[size() - 1],
		curDeltas,
		getStructure(size() - 1),
		batchSize
	);

	for (size_t i = size() - 2; i > 1; i++) {
		
		layerDeltas(
			*gpu,
			*WeightBuffers[i],
			prevDeltas,
			*layerBuffers[i],
			curDeltas,
			getStructure(i + 1),
			getStructure(i),
			batchSize
		);

		weightGradients(
			*gpu,
			*layerBuffers[i],
			prevDeltas,
			weightGradientBuffer,
			getStructure(i),
			getStructure(i+1),
			batchSize
		);

		reduceStridedBatchMajor(*gpu, prevDeltas, batchSize, getStructure(i+1));

		cl::Buffer& biasGradientBuffer = prevDeltas;

		modifyParameters(
			*gpu,
			weightGradientBuffer,
			biasGradientBuffer,
			*WeightBuffers[i],
			*BiasBuffers[i],
			learningRate,
			getStructure(i),
			getStructure(i + 1)
		);

		std::swap(prevDeltas, curDeltas);
	}


	weightGradients(
		*gpu,
		*layerBuffers[0],
		prevDeltas,
		weightGradientBuffer,
		getStructure(0),
		getStructure(1),
		batchSize
	);

	reduceStridedBatchMajor(*gpu, prevDeltas, batchSize, getStructure(1));

	cl::Buffer& biasGradientBuffer = prevDeltas;

	modifyParameters(
		*gpu,
		weightGradientBuffer,
		biasGradientBuffer,
		*WeightBuffers[0],
		*BiasBuffers[0],
		learningRate,
		getStructure(0),
		getStructure(1)
	);

}

// TODO: Split batches if gradients will be too large
// TODO: Proper error handling
TestResult NetworkBackPropGPU::TrainSet(const std::vector<Test>& testSet, double learningRate) {

	static std::unique_ptr<cl::Buffer> expectedBuffer;

	static size_t previousBatchSize = 0;
	
	size_t batchSize = testSet.size();

	std::vector<double> input;
	input.reserve(batchSize * inputLayerSize());

	for (const Test& test : testSet) {
		for (const double& inputValue : test.input.values) {
			input.push_back(inputValue);
		}
	}

	std::vector<double> expected;
	expected.reserve(batchSize * outputLayerSize());

	// TODO: Use insert instead
	for (const Test& test : testSet) {
		for (const double& expectedValue : test.expected.values) {
			expected.push_back(expectedValue);
		}
	}

	if (batchSize == previousBatchSize) {
		// TODO: Disable blocking
		gpu->queue.enqueueWriteBuffer(*layerBuffers[0], CL_TRUE, 0, sizeof(double) * input.size(), input.data());
		gpu->queue.enqueueWriteBuffer(*expectedBuffer, CL_TRUE, 0, sizeof(double) * expected.size(), expected.data());
	}
	else {
		expectedBuffer.reset();
		expectedBuffer = std::make_unique<cl::Buffer>(gpu->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * expected.size(), expected.data());

		layerBuffers[0].reset();
		layerBuffers[0] = std::make_unique<cl::Buffer>(gpu->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * input.size(), input.data());

		for (size_t i = 1; i < this->size(); i++) {
			layerBuffers[i].reset();
			layerBuffers[i] = std::make_unique<cl::Buffer>(gpu->context, CL_MEM_READ_WRITE, sizeof(double) * batchSize * getStructure(i));
		}
	}

	backPropCalculate(batchSize);

	calculateGradients(
		*expectedBuffer,
		learningRate,
		batchSize,
		previousBatchSize
	);

	cl::Buffer& outputBuffer = *layerBuffers[size() - 1];

	double cost = getCost(*gpu, outputBuffer, *expectedBuffer, getStructure(size()-1), batchSize);


	// TODO: Actually calculate the acuracy
	return { cost, 0 };
}

void NetworkBackPropGPU::loadBuffers() {
	for (size_t i = 0; i < parameters.size; i++) {
		// Gets the pointer to the first element in the weights for this layer.
		double* layerWeights = parameters.weights[i][0].data();
		double* layerBiases = parameters.biases[i].data();

		gpu->queue.enqueueReadBuffer(*WeightBuffers[i], CL_TRUE, 0, getStructure(i) * getStructure(i+1), layerWeights);
		gpu->queue.enqueueReadBuffer(*BiasBuffers[i], CL_TRUE, 0, getStructure(i+1), layerBiases);

	}
}

void NetworkBackPropGPU::saveBuffers() {
	// TODO: Write to the buffers instead of clearing
	WeightBuffers.clear();
	BiasBuffers.clear();
	WeightBuffers.reserve(size());
	BiasBuffers.reserve(size());

	for (size_t i = 0; i < parameters.size; i++) {
		// TODO: Just use weightsData
		std::vector<double> layerWeights = flattenVector(parameters.weights[i]);
		std::vector<double> layerBiases(parameters.biases[i].begin(), parameters.biases[i].end());

		cl::Buffer weightBuffer(gpu->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerWeights.size(), layerWeights.data());
		cl::Buffer biasBuffer(gpu->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * layerBiases.size(), layerBiases.data());

		WeightBuffers.push_back(std::make_unique<cl::Buffer>(weightBuffer));
		BiasBuffers.push_back(std::make_unique<cl::Buffer>(biasBuffer));
	}
}

}
