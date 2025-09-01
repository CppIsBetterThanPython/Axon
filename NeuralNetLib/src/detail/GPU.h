#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220

#include <CL/cl2.hpp>

bool initOpenCL();

class GPU {
public:
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	size_t maxWG;
	std::unordered_map<std::string, cl::Kernel> kernels;

	GPU() {
		initOpenCL();

		platform = cl::Platform::getDefault();
		device = cl::Device::getDefault();
		context = cl::Context(device);
		queue = cl::CommandQueue(context, device);
		maxWG = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	}

	void BuildKernel(const std::string& name, const char* source) {
		if (kernels.count(name)) return;

		cl::Program program(context, source);

		program.build();

		cl::Kernel kernel(program, name);

		kernels[name] = kernel;
	}
};