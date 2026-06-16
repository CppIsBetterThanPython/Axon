
#include "gpuUtils.hpp"

void reductionPass(
	GPU& gpu,
	const cl::Buffer& source,
	cl::Buffer& dest,
	size_t source_size
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void reductionPass(
	__global const double* source,
	__global double* dest,
	__local double* localData,
	ulong size
) {
	int globalID = get_global_id(0);
	int localID = get_local_id(0);
	int groupID = get_group_id(0);

	if (globalID < size) {
		localData[localID] = source[globalID];
	} else {
		localData[localID] = 0.0;
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
	kernel.setArg(2, cl::Local(sizeof(double) * localSize));
	kernel.setArg(3, (cl_ulong) source_size);

	cl::NDRange local(localSize);
	cl::NDRange global(globalSize);
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	gpu.queue.finish();
}

double reduce(
	GPU& gpu,
	cl::Buffer buf,
	size_t size
) {
	size_t workGroupSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));

	size_t otherSize = (size - 1) / workGroupSize + 1;

	cl::Buffer other(gpu.context, CL_MEM_READ_WRITE, sizeof(double) * otherSize);

	while (size > 1) {
		reductionPass(gpu, buf, other, size);

		size = (size - 1) / workGroupSize + 1;

		std::swap(buf, other);
	}

	double result;

	gpu.queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(double), &result);

	return result;

}

void stridedReductionPass(
	GPU& gpu,
	const cl::Buffer& source,
	cl::Buffer& dest,
	size_t strideSize,
	size_t batchCount
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void stridedReductionPass(
		__global const double* source,	// The initial values/partial sums that are being reduced
		__global double* dest,		// The final value/parital sums that have been calculated
		__local double* localData,	// Local array that stores the intermediate sums
		ulong strideSize,
		ulong batchCount
) {
	
	size_t strideID = get_global_id(0);
	size_t localStrideID = get_local_id(0);
	size_t groupID = get_group_id(0);
	size_t localStrideSize = get_local_size(0);

	size_t localBatchID = get_local_id(1);
	size_t batchID = get_global_id(1);

	__local double* localStride = localData + localBatchID * localStrideSize;
	__global const double* globalStride = source + batchID * strideSize;

	if (strideID < strideSize && batchID < batchCount) {
		localStride[localStrideID] = globalStride[strideID];
	} else {
		localStride[localStrideID] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (size_t stride = localStrideSize / 2; stride > 0; stride /= 2) {
		if (localStrideID < stride) {
			localStride[localStrideID] += localStride[localStrideID + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localStrideID == 0) {
		dest[batchID * get_num_groups(0) + groupID] = localStride[0];
	}
}
)CLC";

	gpu.BuildKernel("stridedReductionPass", kernelSource);

	cl::Kernel& kernel = gpu.kernels["stridedReductionPass"];

	size_t localStrideSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t localBatchCount = std::min((size_t)256, std::min(gpu.maxWG / localStrideSize, (size_t)1));
	size_t globalStrideSize = ((strideSize + localStrideSize - 1) / localStrideSize) * localStrideSize;
	size_t globalBatchCount = ((batchCount + localBatchCount - 1) / localBatchCount) * localBatchCount;

	size_t workGroupSize = localStrideSize * localBatchCount;

	kernel.setArg(0, source);
	kernel.setArg(1, dest);
	kernel.setArg(2, cl::Local(sizeof(double) * workGroupSize));
	kernel.setArg(3, (cl_ulong)strideSize);
	kernel.setArg(4, (cl_ulong)batchCount);

	cl::NDRange global(globalStrideSize, globalBatchCount);
	cl::NDRange local(localStrideSize, localBatchCount);
	
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	gpu.queue.finish();
}

void reduceStrided(
	GPU& gpu,
	cl::Buffer& buf,
	size_t strideSize,
	size_t batchCount
) {

	size_t localStrideSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t localBatchCount = std::min((size_t)256, std::min(gpu.maxWG / localStrideSize, (size_t)1));

	size_t otherStrideSize = (strideSize - 1) / localStrideSize + 1;

	cl::Buffer other(gpu.context, CL_MEM_READ_WRITE, sizeof(double) * otherStrideSize * batchCount);

	while (strideSize > 1) {
		stridedReductionPass(gpu, buf, other, strideSize, batchCount);

		strideSize = (strideSize - 1) / localStrideSize + 1;

		std::swap(buf, other);
	}
}

void stridedReductionPassBatchMajor(
	GPU& gpu,
	const cl::Buffer& source,
	cl::Buffer& dest,
	size_t strideSize,
	size_t batchCount
) {

	static constexpr const char* kernelSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void stridedReductionBatchMajorPass(
		__global const double* source,	// The initial values/partial sums that are being reduced
		__global double* dest,		// The final value/parital sums that have been calculated
		__local double* localData,	// Local array that stores the intermediate sums
		ulong strideSize,
		ulong batchCount
) {

	size_t strideID = get_global_id(0);
	size_t localStrideID = get_local_id(0);
	size_t groupID = get_group_id(0);
	size_t localStrideSize = get_local_size(0);

	size_t localBatchID = get_local_id(1);
	size_t batchID = get_global_id(1);

	__local double* localStride = localData + localBatchID * localStrideSize;

	if (strideID < strideSize && batchID < batchCount) {
		localStride[localStrideID] = source[strideID * batchCount + batchID];
	} else {
		localStride[localStrideID] = 0.0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (size_t stride = localStrideSize / 2; stride > 0; stride /= 2) {
		if (localStrideID < stride) {
			localStride[localStrideID] += localStride[localStrideID + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (localStrideID == 0) {
		dest[batchID * get_num_groups(0) + groupID] = localStride[0];
	}
}
)CLC";

	gpu.BuildKernel("stridedReductionBatchMajorPass", kernelSource);

	cl::Kernel& kernel = gpu.kernels["stridedReductionBatchMajorPass"];

	size_t localStrideSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t localBatchCount = std::min((size_t)256, std::min(gpu.maxWG / localStrideSize, (size_t)1));
	size_t globalStrideSize = ((strideSize + localStrideSize - 1) / localStrideSize) * localStrideSize;
	size_t globalBatchCount = ((batchCount + localBatchCount - 1) / localBatchCount) * localBatchCount;

	size_t workGroupSize = localStrideSize * localBatchCount;

	kernel.setArg(0, source);
	kernel.setArg(1, dest);
	kernel.setArg(2, cl::Local(sizeof(double) * workGroupSize));
	kernel.setArg(3, strideSize);
	kernel.setArg(4, batchCount);

	cl::NDRange global(globalStrideSize, globalBatchCount);
	cl::NDRange local(localStrideSize, localBatchCount);
	
	gpu.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	gpu.queue.finish();
}

void reduceStridedBatchMajor(
	GPU& gpu,
	cl::Buffer& buf,
	size_t batchCount,
	size_t strideSize
) {

	size_t localStrideSize = std::min((size_t)256, std::bit_floor(gpu.maxWG));
	size_t localBatchCount = std::min((size_t)256, std::min(gpu.maxWG / localStrideSize, (size_t)1));

	size_t otherStrideSize = (strideSize - 1) / localStrideSize + 1;

	cl::Buffer other(gpu.context, CL_MEM_READ_WRITE, sizeof(double) * otherStrideSize * batchCount);

	if (strideSize > 1) {
		stridedReductionPassBatchMajor(gpu, buf, other, strideSize, batchCount);

		strideSize = (strideSize - 1) / localStrideSize + 1;

		std::swap(buf, other);
	}

	while (strideSize > 1) {
		stridedReductionPass(gpu, buf, other, strideSize, batchCount);

		strideSize = (strideSize - 1) / localStrideSize + 1;

		std::swap(buf, other);
	}
}
