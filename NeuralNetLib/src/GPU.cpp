#include "NeuralNet.h"
#include "GPU.h"

using std::vector;

bool initOpenCL() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform plat;
	for (auto& p : platforms) {
		std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if (platver.find("OpenCL 2.") != std::string::npos ||
			platver.find("OpenCL 3.") != std::string::npos) {

			plat = p;
		}
	}
	if (plat() == 0) {
		std::cerr << "No OpenCL 2.0 or newer platform found." << std::endl;
		return 1;
	}

	cl::Platform newP = cl::Platform::setDefault(plat);
	if (newP != plat) {
		std::cerr << "Failed to set default platform." << std::endl;
		return 1;
	}

	std::vector<cl::Device> devices;
	newP.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	if (devices.empty()) {
		std::cerr << "No OpenCL GPU device found." << std::endl;
		return 1;
	}
	cl::Device device = devices.front();

	cl::Device newD = cl::Device::setDefault(device);
	if (newD != device) {
		std::cerr << "Failed to set default device." << std::endl;
		return 1;
	}

	return 0;
}

std::vector<double> multiplyVectorOpenCL(const vector<double>& input, double factor) {
	std::vector<double> data = input;

	const char* kernelSource = R"CLC(
	__kernel void multiply_vector(__global double* vec, double factor) {
		int id = get_global_id(0);
		vec[id] *= factor;
	}
	)CLC";

	static cl::Platform platform = cl::Platform::getDefault();
	static cl::Device device = cl::Device::getDefault();

	cl::Context context(device);
	cl::CommandQueue queue(context, device);

	cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * data.size(), data.data());

	cl::Program program(context, kernelSource);

	program.build({ device });
	cl::Kernel kernel(program, "multiply_vector");

	kernel.setArg(0, buffer);
	kernel.setArg(1, factor);

	cl::NDRange global(data.size());
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	queue.finish();

	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(double) * data.size(), data.data());

	return data;
}

static void calculateLayer(
	GPU& gpu,
	const cl::Buffer& prevNodes,
	const cl::Buffer& weights,
	const cl::Buffer& biases,
	cl::Buffer& nodes,
	size_t prevNodesSize,
	size_t nodesSize) {

	const char* kernelSource = R"CLC(
	#pragma OPENCL EXTENSION cl_khr_fp64 : enablei 
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

void Network::calculateGPU() {
	// Make buffers ping pong
	GPU& gpu = *this->gpu.value();
	cl::Buffer bufferA(gpu.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * largestLayer(), inputLayer->nodeData.data());
	cl::Buffer bufferB(gpu.context, CL_MEM_READ_WRITE, largestPassLayer());

	cl::Buffer* prevLayerBuffer = &bufferA;
	cl::Buffer* curLayerBuffer = &bufferB;

	for (size_t currentLayerIndex = 1; currentLayerIndex < size(); currentLayerIndex++) {
		Layer& prevLayer = (*this)[currentLayerIndex - 1];
		Layer& curLayer = (*this)[currentLayerIndex];

		cl::Buffer& weightsBuffer = *WeightBuffers.value()[currentLayerIndex-1];
		cl::Buffer& biasesBuffer = *BiasBuffers.value()[currentLayerIndex-1];

		calculateLayer(gpu, *prevLayerBuffer, weightsBuffer, biasesBuffer, *curLayerBuffer, prevLayer.size(), curLayer.size());

		std::swap(curLayerBuffer, prevLayerBuffer);
	}
	gpu.queue.enqueueReadBuffer(*prevLayerBuffer, CL_TRUE, 0, sizeof(double) * (*this)[size() - 1].nodeData.size(), (*this)[size() - 1].nodeData.data());
}