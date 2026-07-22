#pragma once

#include <Model.hpp>
#include <graph.hpp>

#include <optional>

class CompilerImpl {
public:
	virtual ~CompilerImpl() = default;

	virtual ModelTemplate compile(Graph graph) = 0;

	virtual constexpr Platform getPlatform() = 0;
};

class CompilerCPU : public CompilerImpl {
public:	

	constexpr Platform getPlatform() override {
		return Platform::CPU;
	}

	ModelTemplate compile(Graph graph) override {

	}
};

class Compiler {
	std::unique_ptr<CompilerImpl> impl;
public:
	Compiler(std::unique_ptr<CompilerImpl>& impl) : impl(std::move(impl)) { }

	static std::optional<Compiler> createCompiler(Platform platform) {
		switch (platform) {
			case Platform::CPU: {
				std::unique_ptr<CompilerImpl> impl = std::make_unique<CompilerCPU>();
				return Compiler(impl);
			}
			case Platform::OpenCL:
				return std::nullopt;
			default:
				return std::nullopt;
		}
	}

	ModelTemplate compile(Graph graph) {
		return impl->compile(graph);
	}
};
