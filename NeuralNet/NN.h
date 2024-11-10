#pragma once

#include <filesystem>

#include "NeuralNet.h"

class NNFile
{
public:
	std::filesystem::path filePath;

	NNFile(std::string filename);

	bool write(Network network);

	bool read(Network& network);

	operator std::string() const { return filePath.string(); }
};