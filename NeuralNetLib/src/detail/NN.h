#pragma once

#include <filesystem>

#include "NeuralNet.h"

class NNFile
{
public:
	std::filesystem::path filePath;

	NNFile(std::string filename);

	bool write(Parameters parameters);

	bool read(Parameters& parameters);

	operator std::string() const { return filePath.string(); }
};