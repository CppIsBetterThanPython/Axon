#include "GPU.hpp"
#include "NetworkBackPropGPU.hpp"

namespace axon {

NetworkBackPropGPU::NetworkBackPropGPU(Parameters& parameters) : NetworkGPU(parameters) {

	layerBuffers.resize(parameters.size);

}

static void average(
	GPU& gpu,
	const cl::Buffer& array,
	size_t arraySize
) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void average(
		__global double* array,
		ulong arraySize) {

	int groupSize = get_enqueued_local_size(0);
	
	int localID = get_local_id(0);
	int groupID = get_group_id(0);

	int viewStart = groupSize * groupID * 2;

	int viewSize = (arraySize - viewStart) % (groupSize * 2);

	__local double[viewSize] view;

	int viewIndex = localID * 2;

	view[viewIndex] = array[viewStart + viewIndex];
	view[viewIndex + 1] = array[viewStart + viewIndex + 1];

	for (int stride = 1; stride < viewSize; stride *= 2) {
		if (localID % stride == 0) {
			view[localID * 2 * stride] += view[(localID * 2 + 1) * stride];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (localID == 0) {
		array[groupID] = view[0] / viewSize;
	}
}
)CLC";

	gpu.BuildKernel("average", kernelSource);

	// TODO make BuildKernel return the kernel
	cl::Kernel& kernel = gpu.kernels["average"];

	kernel.setArg(0, array);
	kernel.setArg(1, (cl_long)(arraySize));

	cl::NDRange range((size_t)(arraySize / 2));
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, range);

	gpu.queue.finish();
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

static TestResult performance(
	GPU& gpu,
	const cl::Buffer& output,
	const cl::Buffer& Expected,
	size_t outputLayerSize,
	size_t batchSize
) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double MSE(double predicted, double expected) {
	return pow(predicted - expected, 2);
}

__kernel void modifyParameters(
		__constant double* predicted,	// Gradients that the parameters want to change by
		__constant double* expected,
		__global double* averageCost,
		ulong outputLayerSize,
		ulong batchSize) {
	
	int neuronID = get_global_id(0);
	int batchID = get_global_id(1);
	
	atomic_add(averageCost, MSE(predicted[outputLayerSize * batchID + neuronID], expected[outputLayerSize * batchID + neuronID]));

	if (batchID == 0 && neuronID == 0) {
		averageCost /= batchSize * outputLayerSize;
	}
}
)CLC";

	gpu.BuildKernel("modifyParameters", kernelSource);

	cl::Kernel& weightKernel = gpu.kernels["modifyParameters"];

	weightKernel.setArg(0, weightGradients);
	weightKernel.setArg(1, weights);
	weightKernel.setArg(2, (cl_double)learningRate);

	cl::NDRange weightRange(nodesSize * prevNodesSize);
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, weightRange);

	cl::Kernel& biasKernel = gpu.kernels["batchAverage"];

	biasKernel.setArg(0, biasGradients);
	biasKernel.setArg(1, biases);
	biasKernel.setArg(2, (cl_double)learningRate);

	cl::NDRange biasRange(nodesSize);
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, biasRange);

	gpu.queue.finish();
}

void inputLayerGradients(
	GPU& gpu,
	const cl::Buffer& correctAnswers,
	const cl::Buffer& nodeGradients,
	const cl::Buffer& nextNodes,
	const cl::Buffer& nodes,
	const cl::Buffer& weightGradients,
	const cl::Buffer& biasGradients,
	size_t nextNodesSize,
	size_t nodesSize,
	size_t batchSize

) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static inline double MSEderivative(double predicted, double expected) {
	return 2 * (predicted - expected)
}

__kernel void inputLayerGradients(
		__global double* correctAnswers,
		__global double* nodeGradients,		// Gradients of nodes in layer L
		__global double* nextNodes,			// Nodes in layer L-1
		__global double* nodes,				// Nodes in layer L
		__global double* weightGradients,	// Weight gradients from layer L-1 to L
		__global double* biasGradients,		// Bias gradients of layer L
		ulong nextNodeSize,
		ulong curNodeSize) {
	
	// ID of the neuron in layer L
	int neuronID = get_global_id(0);
	int batchID = get_global_id(1);

	// CALCULATE NODE GRADIENTS
	//////////////////////////////////////////////////
	
	nodeGradients[batchID * curNodeSize + neuronID] =
		MSEderivative(nodes[batchID * curNodeSize + neuronID], correctAnswers[batchID * curNodeSize * neuronID]);

	// CALCULATE WEIGHT GRADIENTS
	//////////////////////////////////////////////////

	for (int i = 0; i < nextNodeSize; i++) {
		double prevNode = nodes[batchID * curNodeSize + neuronID];
		double node = nextNodes[batchID * nextNodeSize + i];

		double nodeDerivative = sigmoidDerivative(prevNode) * node;

		weightGradients[batchID * nextNodeSize * curNodeSize + nextNodeSize * neuronID + i]
			= nodeDerivative * nodeGradients[batchID * curNodeSize + neuronID];
	}

	// CALCULATE BIAS GRADIENTS
	//////////////////////////////////////////////////

	double node = nodes[batchID * curNodeSize + neuronID];

	double nodeDerivative = sigmoidDerivative(node);

	biasGradients[neuronID] = nodeDerivative * nodeGradients[batchID * curNodeSize + neuronID];
}
)CLC";

	gpu.BuildKernel("layerGradients", kernelSource);

	cl::Kernel& kernel = gpu.kernels["layerGradients"];

	kernel.setArg(0, correctAnswers);
	kernel.setArg(1, nodeGradients);
	kernel.setArg(2, nextNodes);
	kernel.setArg(3, nodes);
	kernel.setArg(4, weightGradients);
	kernel.setArg(5, biasGradients);
	kernel.setArg(6, (cl_long)nextNodesSize);
	kernel.setArg(7, (cl_long)nodesSize);

	cl::NDRange global(nodesSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}

void layerGradients(
	GPU& gpu,
	const cl::Buffer& prevNodeGradients,	// Calculated gradients of nodes in layer L+1
	const cl::Buffer& nodeGradients,		// Gradients of nodes in layer L
	const cl::Buffer& prevNodes,			// Nodes in layer L+1
	const cl::Buffer& nodes,				// Nodes in layer L
	const cl::Buffer& nextNodes,			// Nodes in layer L-1
	const cl::Buffer& weights,				// Weights from layer L to L+1
	const cl::Buffer& weightGradients,		// Weight gradients from layer L-1 to L
	const cl::Buffer& biasGradients,		// Bias gradients of layer L
	size_t nextNodesSize,
	size_t nodesSize,
	size_t prevNodesSize,
	size_t batchSize

) {

	const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static inline double sigmoidDerivative(double sigmoid) {
	return sigmoid * (1 - sigmoid);
}

__kernel void layerGradients(
		__global double* prevNodeGradients, // Calculated gradients of nodes in layer L+1
		__global double* nodeGradients,		// Gradients of nodes in layer L
		__global double* nextNodes,			// Nodes in layer L-1
		__global double* nodes,				// Nodes in layer L
		__global double* prevNodes,			// Nodes in layer L+1
		__global double* weights,			// Weights from layer L to L+1
		__global double* weightGradients,	// Weight gradients from layer L-1 to L
		__global double* biasGradients,		// Bias gradients of layer L
		ulong nextNodeSize,
		ulong curNodeSize,
		ulong prevNodeSize) {
	
	// ID of the neuron in layer L
	int neuronID = get_global_id(0);
	int batchID = get_global_id(1);

	// CALCULATE NODE GRADIENTS
	//////////////////////////////////////////////////
	
	// Possibly some redundent calculations here
	nodeGradients[batchID * curNodeSize + neuronID] = 0.0;

	for (int i = 0; i < prevNodeSize; i++) {
		double prevNode = prevNodes[batchID * prevNodeSize + i];

		double weight = weights[i * curNodeSize + neuronID];

		double nodeDerivative = sigmoidDerivative(prevNode) * weight;

		nodeGradients[batchID * curNodeSize + neuronID] += prevNodeGradients[batchID * prevNodeSize + i] * nodeDerivative;
	}

	// CALCULATE WEIGHT GRADIENTS
	//////////////////////////////////////////////////

	for (int i = 0; i < nextNodeSize; i++) {
		double prevNode = nodes[batchID * curNodeSize + neuronID];
		double node = nextNodes[batchID * nextNodeSize + i];

		double nodeDerivative = sigmoidDerivative(prevNode) * node;

		weightGradients[batchID * nextNodeSize * curNodeSize + nextNodeSize * neuronID + i]
			= nodeDerivative * nodeGradients[batchID * curNodeSize + neuronID];
	}

	// CALCULATE BIAS GRADIENTS
	//////////////////////////////////////////////////

	double node = nodes[batchID * curNodeSize + neuronID];

	double nodeDerivative = sigmoidDerivative(node);

	biasGradients[neuronID] = nodeDerivative * nodeGradients[batchID * curNodeSize + neuronID];
}
)CLC";

	gpu.BuildKernel("layerGradients", kernelSource);

	cl::Kernel& kernel = gpu.kernels["layerGradients"];

	kernel.setArg(0, prevNodeGradients);
	kernel.setArg(1, nodeGradients);
	kernel.setArg(2, nextNodes);
	kernel.setArg(3, nodes);
	kernel.setArg(4, prevNodes);
	kernel.setArg(5, weights);
	kernel.setArg(6, weightGradients);
	kernel.setArg(7, biasGradients);
	kernel.setArg(8, (cl_long)nextNodesSize);
	kernel.setArg(9, (cl_long)nodesSize);
	kernel.setArg(10, (cl_long)prevNodesSize);

	cl::NDRange global(nodesSize, batchSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}

void averageGradients(
	GPU& gpu,
	const cl::Buffer& weightGradients,
	const cl::Buffer& biasGradients,
	size_t nodesSize,
	size_t prevNodesSize, // Prev relative to forward pass
	size_t batchSize

) {

		const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Kernel computes the average of the batched outputs from back propagation
__kernel void batchAverage(
		__global double* array,	// Array of elements with 'batchSize' batches each with 'arraySize' elements
		ulong arraySize,
		ulong batchSize) {
	
	int elementID = get_global_id(0);	// The element being averaged
	int threadID = get_global_id(1);	// Half as many threads as batches

	for (int stride = 1; stride < batchSize; stride *= 2) {
		if (threadID % stride == 0) {
			array[(threadID * 2) * arraySize * stride + elementID] += array[(threadID * 2 + 1) * arraySize * stride + elementID];
		}

		// TODO: Only pause workers in the same batch
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	
	if (threadID == 0) {
		array[weightID] /= batchSize;
	}
}
)CLC";

	gpu.BuildKernel("batchAverage", kernelSource);

	cl::Kernel& weightKernel = gpu.kernels["batchAverage"];

	weightKernel.setArg(0, weightGradients);
	weightKernel.setArg(1, (cl_long) (nodesSize * prevNodesSize));
	weightKernel.setArg(2, (cl_long) batchSize);

	cl::NDRange weightRange(nodesSize * prevNodesSize, (size_t)(batchSize / 2));
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, weightRange);

	cl::Kernel& biasKernel = gpu.kernels["batchAverage"];

	biasKernel.setArg(0, biasGradients);
	biasKernel.setArg(1, (cl_long)nodesSize);
	biasKernel.setArg(2, (cl_long)batchSize);

	cl::NDRange biasRange(nodesSize, (size_t)(batchSize / 2));
	gpu.queue.enqueueNDRangeKernel(weightKernel, cl::NullRange, biasRange);

	gpu.queue.finish();
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

	cl::Kernel& biasKernel = gpu.kernels["batchAverage"];

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
	static std::vector<std::unique_ptr<cl::Buffer>> weightGradientBuffer(size() - 1);
	static std::vector<std::unique_ptr<cl::Buffer>> biasGradientBuffer(size() - 1);
	static std::vector<std::unique_ptr<cl::Buffer>> nodeGradientBuffer(size() - 1);
	
	if (batchSize != prevBatchSize) {
		for (size_t i = 0; i < size() - 1; i++) {
			weightGradientBuffer[i] =
				std::make_unique<cl::Buffer>(
					gpu->context,
					CL_MEM_READ_WRITE,
					sizeof(double) * batchSize * getStructure(i) * getStructure(i + 1)
				);

			biasGradientBuffer[i] =
				std::make_unique<cl::Buffer>(
					gpu->context,
					CL_MEM_READ_WRITE,
					sizeof(double) * batchSize * getStructure(i + 1)
				);

			nodeGradientBuffer[i] =
				std::make_unique<cl::Buffer>(
					gpu->context,
					CL_MEM_READ_WRITE,
					sizeof(double) * batchSize * getStructure(i + 1)
				);
		}

	}

	inputLayerGradients(
		*gpu,
		expectedBuffer,
		*nodeGradientBuffer[size() - 2],
		*layerBuffers[size() - 1],
		*layerBuffers[size() - 2],
		*weightGradientBuffer[size() - 2],
		*biasGradientBuffer[size() - 2],
		getStructure(size() - 2),
		getStructure(size() - 1),
		batchSize
	);

	for (size_t i = size() - 2; i > 0; i++) {
		size_t j = i - 1; // For indexing weights and gradients since they have less elements

		layerGradients(
			*gpu,
			*nodeGradientBuffer[j + 1],
			*nodeGradientBuffer[j],
			*layerBuffers[i + 1],
			*layerBuffers[i],
			*layerBuffers[i - 1],
			*WeightBuffers[j + 1],
			*weightGradientBuffer[j],
			*biasGradientBuffer[j],
			getStructure(i - 1),
			getStructure(i),
			getStructure(i + 1),
			batchSize
		);

		averageGradients(
			*gpu,
			*weightGradientBuffer[j],
			*biasGradientBuffer[j],
			getStructure(i),
			getStructure(i - 1),
			batchSize
		);

		modifyParameters(
			*gpu,
			*weightGradientBuffer[j],
			*biasGradientBuffer[j],
			*WeightBuffers[j],
			*BiasBuffers[j],
			learningRate,
			getStructure(i),
			getStructure(i - 1)
		);
	}
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


}


}