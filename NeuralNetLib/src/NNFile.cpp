#include "pch.h"
#include <fstream>
#include <filesystem>

#include "NeuralNet.h"

using namespace std::filesystem;

static bool isValid(path filePath) {
	if (filePath.extension() != ".nn") {
		std::cerr << "Invalid file extension. Expected \".nn\"" << std::endl;
		return 1;
	}

	else if(!exists(filePath)) {
		std::cerr << "Error: File not found!" << std::endl;
		return 1;
	}

	return 0;
}

bool Network::saveNetwork(const path & filePath) const {
	Parameters parameters = this->GetParameters();

	if (isValid(filePath)) {
		return 1;
	}

	std::ofstream WriteFile(filePath, std::ios::binary | std::ios::trunc);

	if (!WriteFile) {
		std::cerr << "Error: Could not open the file for writing!" << std::endl;
		return 1;
	}

	size_t layerCount = parameters.size + 1;

	WriteFile.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

	for (size_t nodeCount : parameters.structure) {
		WriteFile.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
	}

	for (vector<vector<double>>& layer : parameters.data) {
		for (vector<double> node : layer) {
			for (double parameter : node)
				WriteFile.write(reinterpret_cast<const char*>(&parameter), sizeof(parameter));
		}
	}

	if (WriteFile.fail()) {
		std::cerr << "Error: Writing to the file failed!" << std::endl;
		return 1;
	}

	std::cout << "Successfully wrote to file" << std::endl;

	WriteFile.close();

	return 0;
}

bool Network::getNetwork(Parameters& parameters, const path& filePath) {

	if (isValid(filePath)) {
		return 1;
	}

	std::ifstream ReadFile(filePath, std::ios::binary);

	if (!ReadFile) {
		std::cerr << "Error: Could not open the file for reading!" << std::endl;
		return 1;
	}

	size_t layerCount;

	ReadFile.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

	std::vector<size_t> structure;

	for (size_t i = 0; i < layerCount; i++) {
		size_t nodeCount;
		ReadFile.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
		structure.push_back(nodeCount);
	}

	parameters = Parameters(structure);

	for (vector<vector<double>>& layer : parameters.data) {
		for (vector<double>& node : layer) {
			for (double& parameter : node)
				ReadFile.read(reinterpret_cast<char*>(&parameter), sizeof(parameter));
		}
	}

	// Check if reading weight was successful
	if (ReadFile.eof()) {
		std::cerr << "Error: End of file!" << std::endl;
		return 1;
	}

	// Check if reading bias was successful
	if (ReadFile.fail()) {
		std::cerr << "Error: Failed to read file!" << std::endl;
		return 1;
	}

	std::cout << "Successfully read from file" << std::endl;

	ReadFile.close();

	return 0;
}

bool Network::loadNetwork(const path& filePath) {
	Parameters parameters;
	getNetwork(parameters, filePath);

	this->SetParameters(parameters);

	return 0;
}