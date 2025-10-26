#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220

#include <CL/opencl.hpp>

inline bool initOpenCL() {
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