#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include "NN.h"
#include "NeuralNet.h"

using namespace std::filesystem;

NNFile::NNFile(std::string filename) {
	filename += ".nn";

	this->filePath = filename;

	// Create file if does not exist;
	if (!exists(filePath)) {
		//  Create file
		std::ofstream CreateFile(filename);

		// Check if successful
		if (CreateFile) {
			std::cout << "File created" << std::endl;
		}
		else {
			std::cerr << "Error creating file" << std::endl;
		}

		CreateFile.close();
	}
}

bool NNFile::write(Network network) {

	std::ofstream WriteFile(filePath, std::ios::binary);

	// Check if the file was opened successfully
	if (!WriteFile) { // Alternatively, you can use if (!outFile.is_open())
		std::cerr << "Error: Could not open the file for writing!" << std::endl;
		return 1; // Exit with an error code
	}

	size_t layerCount = network.networkSize;

	WriteFile.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

	for (size_t nodeCount : network.structure) {
		WriteFile.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
	}

	for (Layer layer : network.layers) {
		for (Node node : layer.nodes) {
			for (double weight : node.Weights) {
				WriteFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
			}
			WriteFile.write(reinterpret_cast<const char*>(&node.Bias), sizeof(node.Bias));
		}
	}

	// Check if writing was successful
	if (WriteFile.fail()) {
		std::cerr << "Error: Writing to the file failed!" << std::endl;
		return 1; // Exit with an error code
	}

	std::cout << "Successfully wrote to file" << std::endl;

	WriteFile.close();

	return 0;
}

bool NNFile::read(Network& network) {

	std::ifstream ReadFile(filePath, std::ios::binary);

	// Check if the file was opened successfully
	if (!ReadFile) { // Alternatively, you can use if (!outFile.is_open())
		std::cerr << "Error: Could not open the file for reading!" << std::endl;
		return 1; // Exit with an error code
	}

	size_t layerCount;

	ReadFile.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

	std::vector<size_t> structure;

	for (size_t i = 0; i < layerCount; i++) {
		size_t nodeCount;
		ReadFile.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
		structure.push_back(nodeCount);
	}

	network = Network(structure);

	for (Layer& layer : network.layers) {
		for (Node& node : layer.nodes) {
			for (double& weight : node.Weights)
				ReadFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));

			ReadFile.read(reinterpret_cast<char*>(&node.Bias), sizeof(node.Bias));
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