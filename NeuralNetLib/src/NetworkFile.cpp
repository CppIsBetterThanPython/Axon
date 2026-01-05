#include <fstream>

#include "Network.hpp"
#include "pch.h"

using std::vector;
using namespace std::filesystem;

namespace axon {

const uint32_t fileVersion = 0x0000100;

std::string getFileVersion() {
	std::string version = "v";
	version += fileVersion >> 16;
	version += ".";
	version += (fileVersion >> 8) % 0x00000100;
	version += ".";
	version += fileVersion % 0x00000100;

	return version;
}

enum class NetworkType : uint8_t {
	Network,
	NetworkBackProp
};

enum class FileSections : uint8_t
{
	NetworkMetaData,
	Parameters
};

std::error_code saveNetworkMetaData(const Network& network, std::ofstream& file);

std::error_code saveNetwork(const Network& network, const path& filePath) {
	if (filePath.extension() != ".nn")
		return make_error_code(FileError::IncorrectFileType);

	std::ofstream file(filePath, std::ios::binary | std::ios::trunc);

	// TODO: this isnt doing the right thing
	if (!file) {
		std::error_code ec;

		std::filesystem::status(filePath, ec);

		return ec;
	}

	uint8_t magicNumber[4] = { 'a', 'x', 'o', 'n' };

	file.write(reinterpret_cast<char*>(magicNumber), sizeof(magicNumber));

	uint32_t fileVer = fileVersion;
	file.write(reinterpret_cast<char*>(&fileVer), sizeof(fileVer));

	NetworkType networkType = NetworkType::Network;
	file.write(reinterpret_cast<char*>(&networkType), sizeof(networkType));

	saveNetworkMetaData(network, file);

	saveParameters(network.getNetworkParameters(), file);

	file.close();

	return make_error_code(FileError::Ok);
}

struct NetworkMetaData {
	uint64_t seed;
	uint8_t activationFunction;
	uint8_t outputFunction;
	uint8_t weightInitialisation;
	uint8_t biasInitialisation;
};

std::error_code saveNetworkMetaData(const Network& network, std::ofstream& file) {

	FileSections section = FileSections::NetworkMetaData;
	file.write(reinterpret_cast<char*>(&section), sizeof(section));

	NetworkMetaData metaData = {
		network.getSeed(),
		static_cast<uint8_t>(network.getActivationFunction()),
		static_cast<uint8_t>(network.getOuputFunction()),
		static_cast<uint8_t>(network.weightInitialisation),
		static_cast<uint8_t>(network.biasInitialisation)
	};

	uint64_t metaDataSize = sizeof(metaData);
	file.write(reinterpret_cast<char*>(&metaDataSize), sizeof(metaDataSize));
	file.write(reinterpret_cast<char*>(&metaData), sizeof(metaData));

	return make_error_code(FileError::Ok);
}

std::error_code saveParameters(const Parameters& parameters, std::ofstream& file) {
	
	FileSections section = FileSections::Parameters;
	file.write(reinterpret_cast<char*>(&section), sizeof(section));

	uint64_t parametersSize =
		sizeof(parameters.size) +
		sizeof(parameters.structure) +
		sizeof(parameters.weightsData) +
		sizeof(parameters.biasesData);

	file.write(reinterpret_cast<char*>(&parametersSize), sizeof(parametersSize));

	uint64_t layerCount = parameters.size + 1;

	file.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

	for (uint64_t nodeCount : parameters.structure) {
		file.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
	}

	for (const double weight : parameters.weightsData) {
		file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
	}

	for (const double bias : parameters.biasesData) {
		file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
	}

	return make_error_code(FileError::Ok);
}

std::unique_ptr<Parameters> loadParameters(std::error_code& ec, std::ifstream& file);

std::optional<NetworkMetaData> loadNetworkMetaData(std::error_code& ec, std::ifstream& file);

std::unique_ptr<Network> loadNetwork(std::error_code& ec, std::ifstream& file);

std::unique_ptr<Network> loadNetwork(const path& filePath, std::error_code& ec) {
	ec = make_error_code(FileError::Ok);

	std::ifstream file(filePath, std::ios::binary);

	if (!file) {
		std::filesystem::status(filePath, ec);

		return nullptr;
	}

	std::array<uint8_t, 4> magicNumber = { 'a', 'x', 'o', 'n' };

	std::array<uint8_t, 4> loadedNumber{};

	file.read(reinterpret_cast<char*>(&loadedNumber), sizeof(loadedNumber));

	if (magicNumber != loadedNumber) {
		ec = make_error_code(FileError::CorruptFile);
		return nullptr;
	}

	// TODO: Possibly actually do something with the file version
	uint32_t fileVer;

	file.read(reinterpret_cast<char*>(&fileVer), sizeof(fileVer));

	NetworkType networkType;

	file.read(reinterpret_cast<char*>(&networkType), sizeof(networkType));

	switch (networkType)
	{
	case axon::NetworkType::Network:
	{
		std::unique_ptr<Network> network = loadNetwork(ec, file);
		if (network == nullptr) {
			return nullptr;
		}
		return network;
	}
	case axon::NetworkType::NetworkBackProp:
		break;
	default:
		break;
	}

	return nullptr;

}

std::unique_ptr<Network> loadNetwork(std::error_code& ec, std::ifstream& file) {

	std::optional<NetworkMetaData> metaData;
	std::unique_ptr<Parameters> parameters;

	FileSections section;
	while (file.read(reinterpret_cast<char*>(&section), sizeof(section))) {

		switch (section)
		{
		case axon::FileSections::NetworkMetaData:
		{
			metaData = loadNetworkMetaData(ec, file);
			if (!metaData) {
				return nullptr;
			}
			break;
		}
		case axon::FileSections::Parameters:
			parameters = loadParameters(ec, file);
			if (!parameters) {
				return nullptr;
			}
			break;
		default:
			break;
		}
	}

	if (!metaData || !parameters) {
		ec = make_error_code(FileError::CorruptFile);
		return nullptr;
	}

	// TODO: actually read the metaData
	Network* network = new Network(*parameters);

	network->seed = metaData->seed;

	return std::unique_ptr<Network>(network);
}

std::optional<NetworkMetaData> loadNetworkMetaData(std::error_code& ec, std::ifstream& file) {
	NetworkMetaData metaData;

	uint64_t metaDataSize;

	file.read(reinterpret_cast<char*>(&metaDataSize), sizeof(metaDataSize));

	// TODO: Fix this, it prevents forwards compatibility
	if (metaDataSize < sizeof(NetworkMetaData)) {
		ec = make_error_code(FileError::CorruptFile);
		return std::nullopt;
	}

	file.read(reinterpret_cast<char*>(&metaData), sizeof(metaData));

	file.seekg(metaDataSize - sizeof(metaData), std::ios::cur);

	return metaData;
}

std::unique_ptr<Parameters> loadParameters(std::error_code& ec, std::ifstream& file) {
	uint64_t parametersSize;

	file.read(reinterpret_cast<char*>(&parametersSize), sizeof(parametersSize));

	uint64_t layerCount;

	file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));

	std::vector<size_t> structure;

	for (size_t i = 0; i < layerCount; i++) {
		uint64_t nodeCount;
		file.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
		structure.push_back(nodeCount);
	}

	size_t predictedSectionSize = sizeof(layerCount) + layerCount * sizeof(uint64_t);

	for (size_t layerSize : structure) {
		predictedSectionSize += sizeof(double) * layerSize;
	}

	if (predictedSectionSize != parametersSize) {
		ec = make_error_code(FileError::CorruptFile);
	}

	Parameters parameters = Parameters(structure);

	for (double& weight : parameters.weightsData) {
		file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
	}

	for (double& bias : parameters.biasesData) {
		file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
	}

	// TODO: store this in the file, unsafe as if someone saves an uninitialised network and then loads it will be flagged as initialised
	// I just cba to do it now
	parameters.isInitialised = true;

	return std::make_unique<Parameters>(parameters);
}

}