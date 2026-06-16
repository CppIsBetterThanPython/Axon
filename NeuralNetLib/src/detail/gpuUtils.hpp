
#include "GPU.hpp"


double reduce(
	GPU& gpu,
	cl::Buffer buf,
	size_t size
);

void reduceStrided(
	GPU& gpu,
	cl::Buffer& buf,
	size_t strideSize,
	size_t batchSize
);

void reduceStridedBatchMajor(
	GPU& gpu,
	cl::Buffer& buf,
	size_t batchSize,
	size_t strideSize
);
