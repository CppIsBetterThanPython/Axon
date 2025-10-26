#include <fstream>

#include "NeuralNet.h"

using std::vector;
using namespace std::filesystem;

// TODO: Better error handling
// TODO: Make saving cleaner and more abstract

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

bool saveParameters(const Parameters& parameters, const path & filePath) {

	if (filePath.extension() != ".nn") {
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

	for (const double weight : parameters.weightsData) {
		WriteFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
	}

	for (const double bias : parameters.biasesData) {
		WriteFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
	}

	if (WriteFile.fail()) {
		std::cerr << "Error: Writing to the file failed!" << std::endl;
		return 1;
	}

	std::cout << "Successfully wrote to file" << std::endl;

	WriteFile.close();

	return 0;
}

// TODO: Throw errors or smth, or make this a constructor for parameters ig
Parameters getParameters(const path& filePath) {

	if (isValid(filePath)) {
		//return 1;
	}

	std::ifstream ReadFile(filePath, std::ios::binary);

	if (!ReadFile) {
		std::cerr << "Error: Could not open the file for reading!" << std::endl;
		//return 1;
	}

	size_t layerCount;

	ReadFile.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

	std::vector<size_t> structure;

	for (size_t i = 0; i < layerCount; i++) {
		size_t nodeCount;
		ReadFile.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
		structure.push_back(nodeCount);
	}

	Parameters parameters = Parameters(structure);

	for (double& weight : parameters.weightsData) {
		ReadFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
	}

	for (double& bias : parameters.biasesData) {
		ReadFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
	}

	// Check if reading weight was successful
	if (ReadFile.eof()) {
		std::cerr << "Error: End of file!" << std::endl;
		//return 1;
	}

	// Check if reading bias was successful
	if (ReadFile.fail()) {
		std::cerr << "Error: Failed to read file!" << std::endl;
		//return 1;
	}

	std::cout << "Successfully read from file" << std::endl;

	ReadFile.close();

	//return 0;

	return parameters;
}

bool Network::loadNetwork(const path& filePath) {
	parameters = getParameters(filePath);

	return 0;
}

bool Network::saveNetwork(const std::filesystem::path& filename) const {
	saveParameters(parameters, filename);

	return 0;
}