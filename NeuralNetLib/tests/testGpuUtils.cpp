#include <CL/cl.h>
#include <gtest/gtest.h>

#include <random>

#include "../src/detail/gpuUtils.hpp"

constexpr double epsilon = 1e-16;

std::unique_ptr<GPU> gpu = std::make_unique<GPU>();

std::vector<double> genVec(size_t size) {
	static std::mt19937 gen(1234);
	static std::uniform_real_distribution<double> dist(-10, 10);

	std::vector<double> vec;
	vec.reserve(size);

	for (size_t i = 0; i < size; i++) {
		vec.push_back(dist(gen));
	}

	return vec;
}

double reduceCPU(const std::vector<double>& vec) {
	double sum = 0.0f;

	for (const double elem : vec) {
		sum += elem;
	}

	return sum;
}

std::vector<double> reduceStridedCPU(const std::vector<double>& vec, const size_t stride) {
	size_t batchSize = vec.size() / stride;

	std::vector<double> result;
	result.reserve(batchSize);

	for (size_t i = 0; i < batchSize; i++) {
		double sum = 0.0;
		for (size_t j = 0; j < stride; j++) {
			sum += vec[i * stride + j];
		}
		result.push_back(sum);
	}

	return result;
}

std::vector<double> reduceStridedBatchMajorCPU(const std::vector<double>& vec, const size_t stride) {
	size_t batchSize = vec.size() / stride;

	std::vector<double> result;
	result.reserve(batchSize);

	for (size_t i = 0; i < batchSize; i++) {
		double sum = 0.0;
		for (size_t j = 0; j < stride; j++) {
			sum += vec[j * batchSize + i];
		}
		result.push_back(sum);
	}

	return result;
}

double reduceWrapper(std::vector<double> vec) {
	cl::Buffer buf(
		gpu->context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * vec.size(),
		vec.data()
	);

	return reduce(*gpu, buf, vec.size());
}

std::vector<double> reduceStridedWrapper(std::vector<double>& vec, const size_t stride) {
	cl::Buffer buf(
		gpu->context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * vec.size(),
		vec.data()
	);

	size_t batchSize = vec.size() / stride;

	reduceStrided(*gpu, buf, stride, batchSize);

	std::vector<double> output;
	output.resize(batchSize);

	gpu->queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(double) * batchSize, output.data());

	return output;
}

std::vector<double> reduceStridedBatchMajorWrapper(std::vector<double>& vec, const size_t stride) {
	cl::Buffer buf(
		gpu->context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * vec.size(),
		vec.data()
	);

	size_t batchSize = vec.size() / stride;

	reduceStridedBatchMajor(*gpu, buf, batchSize, stride);

	std::vector<double> output;
	output.resize(batchSize);

	gpu->queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(double) * batchSize, output.data());

	return output;
}

TEST(ReduceTest, SmallArray) {
	std::vector<double> vec = {1.0, 2.0, 4.0, 2.4, 2.5, 1.6};

	double cpuResult = reduceCPU(vec);
	double gpuResult = reduceWrapper(vec);

	EXPECT_DOUBLE_EQ(cpuResult, gpuResult);
}


TEST(ReduceTest, LargeArray) {
	std::vector<double> vec = genVec(1000);

	double cpuResult = reduceCPU(vec);
	double gpuResult = reduceWrapper(vec);

	EXPECT_DOUBLE_EQ(cpuResult, gpuResult);
	
}

TEST(ReduceStridedTest, SmallArray) {
	std::vector<double> vec = {1.0, 2.0, 4.0, 2.4, 2.5, 1.6, 1.4, 7.3, 8.4, 5.8, 6.2, 9.1};

	size_t stride = 3;

	std::vector<double> cpuResult = reduceStridedCPU(vec, stride);
	std::vector<double> gpuResult = reduceStridedWrapper(vec, stride);

	ASSERT_EQ(cpuResult.size(), gpuResult.size());

	for (size_t i = 0; i < gpuResult.size(); ++i) {
	    EXPECT_DOUBLE_EQ(gpuResult[i], cpuResult[i])
		<< "Mismatch at index " << i;
	}

}

TEST(ReduceStridedTest, LargeArray) {
	std::vector<double> vec = genVec(10000);

	size_t stride = 1000;

	std::vector<double> cpuResult = reduceStridedCPU(vec, stride);
	std::vector<double> gpuResult = reduceStridedWrapper(vec, stride);

	ASSERT_EQ(cpuResult.size(), gpuResult.size());

	for (size_t i = 0; i < gpuResult.size(); ++i) {
	    EXPECT_NEAR(gpuResult[i], cpuResult[i], 1e-10)
		<< "Mismatch at index " << i;
	}

}

TEST(ReduceStridedBatchMajorTest, SmallArray) {
	std::vector<double> vec = {1.0, 2.0, 4.0, 2.4, 2.5, 1.6, 1.4, 7.3, 8.4, 5.8, 6.2, 9.1};

	size_t stride = 3;

	std::vector<double> cpuResult = reduceStridedBatchMajorCPU(vec, stride);
	std::vector<double> gpuResult = reduceStridedBatchMajorWrapper(vec, stride);

	ASSERT_EQ(cpuResult.size(), gpuResult.size());

	for (size_t i = 0; i < gpuResult.size(); ++i) {
	    EXPECT_DOUBLE_EQ(gpuResult[i], cpuResult[i])
		<< "Mismatch at index " << i;
	}

}

TEST(ReduceStridedBatchMajorTest, LargeArray) {
	std::vector<double> vec = genVec(10000);

	size_t stride = 1000;

	std::vector<double> cpuResult = reduceStridedBatchMajorCPU(vec, stride);
	std::vector<double> gpuResult = reduceStridedBatchMajorWrapper(vec, stride);

	ASSERT_EQ(cpuResult.size(), gpuResult.size());

	for (size_t i = 0; i < gpuResult.size(); ++i) {
	    EXPECT_NEAR(gpuResult[i], cpuResult[i], 1e-10)
		<< "Mismatch at index " << i;
	}

}

