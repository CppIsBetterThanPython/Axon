#pragma once

#include <Model.hpp>
#include <graph.hpp>

class CompilerImpl {
public:
	virtual ~CompilerImpl() = default;

	virtual ModelTemplate compile(Graph graph) = 0;

	virtual constexpr Platform getPlatform() = 0;
}

class CompilerCPU : CompilerImpl {
public:
	constexpr Platform getPlatform() override {
		return Platform::CPU;
	}

	ModelTemplate compiler(Graph graph) override {
		graph
	}
}

class Compiler {
	std::unique_ptr<CompilerImpl> impl;
public:
	Compiler(CompilerImpl impl) : impl(impl) {
	}

	static std::optional<Compiler> createCompiler(Platform platform)

	ModelTemplate compile(Graph graph) {
		return impl->compile(graph);
	}
}
