#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <random>

#include <bit>

#include "GPU.hpp"

std::vector<float> genVec(size_t size) {
	static std::mt19937 gen(1234);
	static std::uniform_real_distribution<float> dist(-10, 10);

	std::vector<float> vec;
	vec.reserve(size);

	for (size_t i = 0; i < size; i++) {
		vec.push_back(dist(gen));
	}

	return vec;
}

void vectorMul(
	GPU& gpu,
	const cl::Buffer& vecA,
	const cl::Buffer& vecB,
	cl::Buffer& vecOut,
	size_t vec_size
) {

	static constexpr const char* kernelSource = R"CLC(
__kernel void vectorMul(
	__global const float* vecA,
	__global const float* vecB,
	__global float* vecOut
) {
	int itemID = get_global_id(0);

	vecOut[itemID] = vecA[itemID] * vecB[itemID];
}
)CLC";

	gpu.BuildKernel("vectorMul", kernelSource);

	cl::Kernel& kernel = gpu.kernels["vectorMul"];

	kernel.setArg(0, vecA);
	kernel.setArg(1, vecB);
	kernel.setArg(2, vecOut);

	cl::NDRange global(vec_size);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
	gpu.queue.finish();
}

void reductionPass(
	GPU& gpu,
	const cl::Buffer& source,
	cl::Buffer& dest,
	size_t source_size
) {

	static constexpr const char* kernelSource = R"CLC(
__kernel void reductionPass(
	__global const float* source,
	__global float* dest,
	__local float* localData,
	ulong size
) {
	int globalID = get_global_id(0);
	int localID = get_local_id(0);
	int groupID = get_group_id(0);

	if (globalID < size) {
		localData[localID] = source[globalID];
	} else {
		localData[localID] = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (ulong stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
		if (localID < stride) {
			localData[localID] += localData[localID + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localID == 0) {
		dest[groupID] = localData[0];
	}
}
)CLC";
	
	gpu.BuildKernel("reductionPass", kernelSource);

	cl::Kernel& kernel = gpu.kernels["reductionPass"];

	size_t localSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t globalSize = ((source_size + localSize - 1) / localSize) * localSize;

	kernel.setArg(0, source);
	kernel.setArg(1, dest);
	kernel.setArg(2, cl::Local(sizeof(float) * localSize));
	kernel.setArg(3, (cl_ulong) source_size);

	cl::NDRange local(localSize);
	cl::NDRange global(globalSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	gpu.queue.finish();
}

float reduce(
	GPU& gpu,
	cl::Buffer buf,
	size_t size
) {
	size_t workGroupSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));

	size_t otherSize = (size - 1) / workGroupSize + 1;

	cl::Buffer other(gpu.context, CL_MEM_READ_WRITE, sizeof(float) * otherSize);

	while (size > 1) {
		reductionPass(gpu, buf, other, size);

		size = (size - 1) / workGroupSize + 1;

		std::swap(buf, other);
	}

	float result;

	gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(float), &result);

	return result;

}

float reduceCPU(const std::vector<float>& vec) {
	float sum = 0.0f;

	for (const float elem : vec) {
		sum += elem;
	}

	return sum;
}



int main() {
	std::vector<float> vec = genVec(1000);

	std::unique_ptr<GPU> gpu = std::make_unique<GPU>();

	std::vector<float> a = {1.0, 2.0, 4.0, 2.4, 2.5, 1.6};

	std::unique_ptr<cl::Buffer> bufA = std::make_unique<cl::Buffer>(
		gpu->context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * vec.size(),
		vec.data()
	);

	std::cout << reduce(*gpu, *bufA, vec.size()) << std::endl;

	std::cout << reduceCPU(vec) << std::endl;
}
