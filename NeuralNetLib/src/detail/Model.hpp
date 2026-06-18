#pragma once

#include <functional>

#include<graph.hpp>

enum class StorageType {
	RAM,
	OpenCL_Buffer
};

enum class Platform {
	CPU,
	OpenCL
};

class Storage {
public:
	virtual ~Storage() = default;

	virtual void* data() = 0;
	virtual size_t bytes() const = 0;
};

class TensorBuffer {
	TensorInfo metaData;

private:
	StorageType type;
	std::unique_ptr<Storage> data;
};

class ModelTemplate {
	std::unordered_map<std::string, TensorInfo> inputs;
	std::unordered_map<std::string, TensorInfo> outputs;

	std::vector<TensorInfo> parameters;

	std::function<std::vector<TensorBuffer> (std::vector<TensorBuffer>, std::vector<TensorBuffer>)> forwardPass();

	Platform platform;
};

class Model {
	std::vector<TensorBuffer> input;
	std::vector<TensorBuffer> outputs;
	
	std::vector<TensorBuffer> parameters;

	Platform platform;

	std::function<std::vector<TensorBuffer> (std::vector<TensorBuffer>, std::vector<TensorBuffer>)> forwardPass();

	
};
