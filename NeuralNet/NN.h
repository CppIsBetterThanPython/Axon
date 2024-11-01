#pragma once

#include <iostream>

#include "NeuralNet.h"

class NNFile
{
public:
	std::string filename;

	NNFile(std::string filename);

	bool write(Network network);

	bool read(Network& network);

	operator std::string() const { return filename; }
};