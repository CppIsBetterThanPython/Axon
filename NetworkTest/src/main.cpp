#include <algorithm>  // For std::sort
#include <chrono>
#include <windows.h>
#include <functional>
#include <unordered_map>
#include <map>
#include <any>
#include <fstream>
#include <cstdlib>

#include "NetworkBackProp.h"
#include "Mnist.h"

using nbp = NetworkBackProp;
namespace fs = std::filesystem;

const fs::path networksPath = std::getenv("LOCALAPPDATA") + std::string("\\neuralNetTraining\\networks");
const fs::path autoSavesPath = networksPath / "autoSaves";
const fs::path downloadsPath = std::getenv("userprofile") + std::string("\\Downloads");

bool autoSave = true;
int saveFrequency = 100;
double learningRate = 0.1;

inline bool IsInRange(double num, double lower, double upper) {
    return (num >= lower && num <= upper);
}

void init() {
    srand(static_cast<int>(time(NULL)));

    if (!exists(autoSavesPath)) {
        fs::create_directories(autoSavesPath);
    }
}

vector<nbp::Test> generateTestSet(int size) {
    vector<nbp::Test> testSet(size);

    for (int i = 0; i < size; i++) {
        vector<double> x = { static_cast<double>(rand() % 100) + RandomReal(), static_cast<double>(rand() % 100) + RandomReal() };
        vector<double> y = { static_cast<double>(rand() % 100) + RandomReal(), static_cast<double>(rand() % 100) + RandomReal() };

        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());

        vector<double> point = { static_cast<double>(rand() % 100) + RandomReal(), static_cast<double>(rand() % 100) + RandomReal() };

        vector<double> input = { x[0], y[0], x[1], y[1], point[0], point[1] };

        nbp::Answer answer(2, 0);

        if (IsInRange(point[0], x[0], x[1]) && IsInRange(point[1], y[0], y[1])) {
            answer[0] = 1;
        }
        else {
            answer[1] = 1;
        }

        nbp::Test TestData = { input, answer };

        testSet[i] = TestData;
    }

    return testSet;
}

// Smoothing factor closer to 1 values newer values more, 0 values older values more
double EWMA(double pastAverage, double currentCost, double smoothingFactor, size_t time) {
    double newAverage = smoothingFactor * pastAverage + (1 - smoothingFactor) * currentCost;

    newAverage /= 1 - pow(smoothingFactor, time);

    return newAverage;
};

bool create_file(fs::path filePath) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        file.close();
        std::cerr << "Failed to create file" << std::endl;
        return 1;
    }
    file.close();
    return 0;
}

static void autoSaver(const nbp& network, size_t code, double cost) {
    fs::path filePath = autoSavesPath / (std::to_string(code) + "-" + std::to_string(cost) + ".nn");

    if (!exists(filePath)) {
        create_file(filePath);
    }

    network.saveNetwork(filePath);
}

void train(NetworkBackProp& network, const size_t& duration, const size_t& batchSize, const size_t& valSize, const size_t& valBatchSize, const size_t& valFrequency = 1) {
    static vector<nbp::Test> fullTrainSet = getTrainSet();
    if (valSize > fullTrainSet.size())
        throw std::invalid_argument("Validation Set too large");

    static vector<nbp::Test> trainSet(fullTrainSet.begin(), fullTrainSet.end() - valSize);
    static vector<nbp::Test> valSet(fullTrainSet.end() - valSize, fullTrainSet.end());

    std::vector<nbp::Test> trainBatch(trainSet.begin(), trainSet.begin() +  batchSize);

    double currentCost = 1;
    double weightedCost = network.trainNetworkBackPropogation(trainBatch, 1).first;

    for (size_t i = 0; i < duration; i++) {
        size_t batchBegin = (i * batchSize) % (trainSet.size() - batchSize);
        trainBatch = std::vector<nbp::Test>(trainSet.begin() + batchBegin, trainSet.begin() + batchBegin + batchSize);

        network.trainNetworkBackPropogation(trainBatch, learningRate);

        if (i % valFrequency == 0) {
            size_t batchBegin = (i * valBatchSize) % max(valSet.size() - valBatchSize, 1);
            std::vector<nbp::Test> valBatch(trainSet.begin() + batchBegin, trainSet.begin() + batchBegin + valBatchSize);
            auto [cost, accuracy] = network.testNetworkBackPropogation(valBatch);

            currentCost = cost;
            weightedCost = EWMA(weightedCost, currentCost, 0.8, i + 1);

            std::cout << i << ":" << std::endl;
            std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
            std::cout << "cost: " << weightedCost << std::endl;

            if (i % saveFrequency == 0 && autoSave) {
                autoSaver(network, i / saveFrequency, weightedCost);
            }
        }
    }
}

/*vector<double> testNetworkLearningSpeed(int testLength, int upperBound, double targetCost, NetworkBackProp& network) {
    int amountCompleted = testLength;

    double averageCount = 0.0;
    double averageElapsed = 0.0;

    for (int i = 0; i < testLength; i++) {
        double currentCost = 1;
        double weightedCost = constantLearningApproach(network).first;

        int instanceCounter = 0;

        // Start time measurement
        auto start = std::chrono::high_resolution_clock::now();

        while (weightedCost > targetCost && instanceCounter <= upperBound) {
            std::cout << instanceCounter << ":" << std::endl;
            auto [cost, accuracy] = constantLearningApproach(network);
            currentCost = cost;
            weightedCost = EWMA(weightedCost, currentCost, 0.8, instanceCounter + 1);

            std::cout << "Accuracy: " << accuracy<< std::endl;
            std::cout << "cost: " << weightedCost << std::endl;

            instanceCounter++;
        }

        // End time measurement
        auto end = std::chrono::high_resolution_clock::now();

        if (instanceCounter > upperBound) {
            amountCompleted--;
        }
        else {
            // Calculate elapsed time
            std::chrono::duration<double> elapsed = end - start;

            averageElapsed += elapsed.count();
            averageCount += instanceCounter;
        }

        network.resetWeights();
    }

    averageCount /= amountCompleted;
    averageElapsed /= amountCompleted;

    std::cout << "Average count: " << averageCount << std::endl;

    std::cout << "Average elapsed time: " << averageElapsed << " seconds\n";

    std::cout << "Amount completed: " << amountCompleted << std::endl;

    return { averageCount, averageElapsed };
}*/

using CmdFunc = std::function<void(const std::vector<std::string>& args)>;

using Cmd = std::pair<std::string, std::pair<CmdFunc, vector<std::string>>>;

bool canCastInt(std::string str) {
    try {
        size_t pos;
        int num = stoi(str, &pos);
        if (pos != str.size()) {
            return false;
        }
        return true;
    }
    catch (std::invalid_argument) {
        return false;
    }
    catch (std::out_of_range) {
        return false;
    }
}

bool canCastBool(std::string str) {
    if (canCastInt(str) && (stoi(str) == 0 || stoi(str) == 1))
        return true;
    else if (str == "FALSE" || str == "TRUE")
        return true;
    else
        return false;
}

bool canCastDouble(std::string str) {
    try {
        size_t pos;
        double num = stod(str, &pos);
        if (pos != str.size()) {
            return false;
        }
        return true;
    }
    catch (std::invalid_argument) {
        return false;
    }
    catch (std::out_of_range) {
        return false;
    }
}

bool stob(const std::string& str) {
    if (!canCastBool(str))
        throw std::invalid_argument("Cannot cast string to bool. String: " + str);
    
    if (str == "FALSE")
        return false;

    else if (str == "TRUE")
        return true;

    else
        return stoi(str);
}

void invalidParams() {
    std::cout << "Ivalid Paramaters" << std::endl;
}

class Menu {
private:
    static std::vector<std::string> splitString(std::string str) {
        std::vector<std::string> output;
        size_t wordBegin = 0;

        for (size_t i = 0; i < str.size(); i++) {
            if (str[i] == ' ') {
                output.push_back(str.substr(wordBegin, max(i - wordBegin, 0)));
                wordBegin = min(i + 1, str.size() - 1);
            }
        }
        output.push_back(str.substr(wordBegin, max(str.size(), 0)));

        return output;
    }

public:
    std::map < std::string, std::pair<CmdFunc, std::vector<std::string>> > commands;

    const CmdFunc Help = [this](const std::vector<std::string> null) {
        if (null.size() != 0 && null.size() != 1) {
            invalidParams();
            return;
        }

        if (null.size() == 0) {
            for (auto& [name, entry] : commands) {
                auto& [func, descs] = entry;

                for (std::string desc : descs)
                    std::cout << name << desc << std::endl;
            }
        }
        else {
            const std::string& commandName = null[0];
            if (!commands.count(commandName)) {
                std::cout << "Command not found" << std::endl;
            }
            for (std::string desc : commands[commandName].second)
                std::cout << commandName << " - " << desc << std::endl;
        }
    };

    Menu() {
        this->commands["help"] = { Help, {
            "Prints all possible commands in current menu.",
            " [command] - Prints the description for the given command."
        } };
    }

    Menu(vector<Cmd> commands) {
        this->commands["help"] = { Help, {
            "Prints all possible commands in current menu.",
            " [command] - Prints the description for the given command."
        } };

        for (auto& [name, entry] : commands) {
            this->commands[name] = entry;
        }
    }

    void getInput() {
        std::string input;
        std::cout << ">> ";
        std::getline(std::cin, input);

        std::vector<std::string> parsedInput = splitString(input);

        std::string funcKey = parsedInput[0];

        parsedInput.erase(parsedInput.begin());

        if (!commands.count(funcKey)) {
            std::cout << "Command not found. Try typing \"help\" for a list of commands." << std::endl;
            return;
        }
        commands[funcKey].first(parsedInput);
    }
};

void printAny(std::any var) {

    if (var.type() == typeid(double*))
        std::cout << *std::any_cast<double*>(var) << std::endl;

    else if (var.type() == typeid(std::string*))
        std::cout << *std::any_cast<std::string*>(var) << std::endl;

    else if (var.type() == typeid(int*))
        std::cout << *std::any_cast<int*>(var) << std::endl;

    else if (var.type() == typeid(bool*))
        if (*std::any_cast<bool*>(var))
            std::cout << "TRUE" << std::endl;
        else
            std::cout << "FALSE" << std::endl;

    else
        std::cout << "Unknown type" << std::endl;
}

int main() {
    init();

    std::unordered_map<std::string, nbp*> Networks;

    std::unordered_map<std::string, std::any> globalParameters = {
        {"autoSave", &autoSave},
        {"saveFreq", &saveFrequency},
        {"learningRate", &learningRate}
    };
    // Set the close handler
    /*if (!SetConsoleCtrlHandler(ConsoleCloseHandler, TRUE)) {
        std::cerr << "Failed to set control handler!\n";
        return 1;
    }*/

    Menu mainMenu;

    CmdFunc get = [&globalParameters](const std::vector<std::string> param) {
        if (param.size() != 1) {
            invalidParams();
            return;
        }
    
        if (!globalParameters.count(param[0])) {
            std::cout << "Parameter does not exist" << std::endl;
            return;
        }
    
        printAny(globalParameters[param[0]]);
        };
    
    CmdFunc getGlobalParameters = [&globalParameters](const std::vector<std::string> param) {
        if (param.size() != 0) {
            invalidParams();
            return;
        }
    
        for (auto& [key, param] : globalParameters) {
            std::cout << key << " - ";
            printAny(param);
        }
        };
    
    CmdFunc set = [&globalParameters](const std::vector<std::string> input) {
        if (input.size() != 2) {
            invalidParams();
            return;
        }
    
        const std::string& param = input[0],
            val = input[1];
    
        if (!globalParameters.count(param)) {
            std::cout << "Parameter does not exist" << std::endl;
            return;
        }
    
        std::any var = globalParameters[param];
    
        if (var.type() == typeid(double*) && canCastDouble(val))
            *std::any_cast<double*>(var) = stod(val);
    
        else if (var.type() == typeid(int*) && canCastInt(val))
            *std::any_cast<int*>(var) = stoi(val);
    
        else if (var.type() == typeid(std::string*))
            *std::any_cast<std::string*>(var) = val;
    
        else if (var.type() == typeid(bool*) && canCastBool(val))
            *std::any_cast<bool*>(var) = stob(val);
    
        else
            std::cout << "Invalid Value" << std::endl;
        };
    
    CmdFunc createNetwork = [&Networks](const std::vector<std::string> input) {
        if (input.size() < 2) {
            invalidParams();
            return;
        }
    
        vector<size_t> networkSize;
        networkSize.reserve(input.size() - 2);
    
        std::vector<std::string> strParams(input.begin() + 2, input.end());
    
        for (std::string param : strParams) {
            if (!canCastInt(param)) {
                invalidParams();
                return;
            }
            networkSize.push_back(stoi(param));
        }
    
        const std::string& netType = input[0];
        const std::string netName = input[1];
        if (netType == "BACK_PROP") {
            if (Networks.count(netName)) {
                std::cout << "Network already exists." << std::endl;
                return;
            }
            Networks[netName] = new nbp(networkSize);
        }
    };
    
    CmdFunc trainNetwork = [&Networks](const std::vector<std::string> input) {
        if (input.size() != 5) {
            invalidParams();
            return;
        }
    
        const std::string& networkName = input[0];
        if (!Networks.count(networkName)) {
            std::cout << "No such network exists." << std::endl;
            return;
        }
    
        vector<int> params;
        params.reserve(input.size() - 1);
    
        std::vector<std::string> strParams(input.begin() + 1, input.end());
    
        for (std::string param : strParams) {
            if (!canCastInt(param)) {
                invalidParams();
                return;
            }
            params.push_back(stoi(param));
        }
    
        if (nbp* network = dynamic_cast<nbp*>(Networks[networkName])) {
            train(*network, params[0], params[1], params[2], params[3]);
        }
        else {
            std::cout << "Incorrect type of network." << std::endl;
            return;
        }
    };

    CmdFunc save = [&Networks](const std::vector<std::string> input) {
        if (input.size() != 2) {
            invalidParams();
            return;
        }

        const std::string& networkName = input[0];
        if (!Networks.count(networkName)) {
            std::cout << "No such network exists." << std::endl;
            return;
        }

        nbp* network = Networks[networkName];

        fs::path path = networksPath / input[1];
        if (path.extension() != ".nn") {
            path += ".nn";
        }

        if (!exists(path)) {
            create_file(path);
        }

        network->saveNetwork(path);
        
    };

    CmdFunc load = [&Networks](const std::vector<std::string> input) {
        if (input.size() != 2) {
            invalidParams();
            return;
        }

        fs::path path = networksPath / input[1];

        if (path.extension() != ".nn") {
            std::cerr << "Incorrect file extension. Expected \".nn\" but got \"" + path.extension().string() + '"';
            return;
        }

        const std::string& networkName = input[0];
        if (!Networks.count(networkName)) {
            Networks[networkName] = new nbp(path);
        }
        else {
            Networks[networkName]->loadNetwork(path);
        }
    };

    CmdFunc del = [&Networks](const std::vector<std::string> input) {
        if (input.size() != 1) {
            invalidParams();
            return;
        }

        const std::string& networkName = input[0];
        if (!Networks.count(networkName)) {
            std::cout << "No such network exists." << std::endl;
            return;
        }

        Networks.erase(networkName);

        };

    CmdFunc list = [&Networks](const std::vector<std::string> input) {
        if (input.size() != 0) {
            invalidParams();
            return;
        }

        if (Networks.size() == 0) {
            std::cout << "No Networks currently loaded." << std::endl;
        }

        for (auto& [key, network] : Networks) {
            std::cout << key << " - { ";
            for (size_t size : network->structure) {
                std::cout << size << " ";
            }
            std::cout << " }" << std::endl;
        }

        };

    CmdFunc listSaved = [](const std::vector<std::string> input) {
        if (input.size() != 0) {
            invalidParams();
            return;
        }

        for (const auto& entry : fs::directory_iterator(networksPath)) {
            fs::path path = entry;
            if (path.extension() == ".nn")
                std::cout << path.filename() << std::endl;
        }

    };

    CmdFunc listAutoSaved = [](const std::vector<std::string> input) {
        if (input.size() != 0) {
            invalidParams();
            return;
        }

        for (const auto& entry : fs::directory_iterator(autoSavesPath)) {
            fs::path path = entry;
            if (path.extension() == ".nn")
                std::cout << path.filename() << std::endl;
        }

        };

    CmdFunc deleteSaved = [](const std::vector<std::string> input) {
        if (input.size() != 1) {
            invalidParams();
            return;
        }

        fs::path networkPath = networksPath / input[0];
        if (!exists(networkPath)) {
            std::cerr << "File not found." << std::endl;
            return;
        }
        if (networkPath.extension() != ".nn") {
            std::cerr << "Incorrect file extention" << std::endl;
            return;
        }

        fs::remove(networkPath);

        };

    CmdFunc exp = [](const std::vector<std::string> input) {
        if (input.size() != 1 && input.size() != 2) {
            invalidParams();
            return;
        }

        fs::path networkPath = networksPath / input[0];
        if (!exists(networkPath)) {
            std::cerr << "File not found." << std::endl;
            return;
        }
        if (networkPath.extension() != ".nn") {
            std::cerr << "Incorrect file extention" << std::endl;
            return;
        }

        fs::path destinationPath;
        if (input.size() == 1) {
            destinationPath = downloadsPath / networkPath.filename();
        }
        else {
            fs::path destinationDir = input[1];
            if (!exists(destinationDir)) {
                std::cerr << "Destination directory not found.";
                return;
            }
            destinationPath = input[1] / networkPath.filename();
        }
        
        fs::copy_file(networkPath, destinationPath);

        };
    
    mainMenu.commands["get"]         = { get,                 { " [param] - Gets the state of a global parameter." } };
    mainMenu.commands["getall"]      = { getGlobalParameters, { " - Gets the state of all global parameters." } };
    mainMenu.commands["set"]         = { set,                 { " [param] [value] - Sets the state of a global parameter." } };
    mainMenu.commands["create"]      = { createNetwork,       { " [type] [name] {size} - Creates a network of the given type." } };
    mainMenu.commands["delete"]      = { del,                 { " [network] - Deletes given Network." } };
    mainMenu.commands["train"]       = { trainNetwork,        { " [network] [duration] [batchSize] [valSize] [valBatchSize] - Trains give Network." } };
    mainMenu.commands["save"]        = { save,                { " [network] [path] - Saves given Network." } };
    mainMenu.commands["load"]        = { load,                { " [network] [path] - Loads given Network." } };
    mainMenu.commands["list"]        = { list,                { " - Lists all loaded Networks." } };
    mainMenu.commands["listSaved"]   = { listSaved,           { " - Lists all saved Networks." } };
    mainMenu.commands["listAuto"]    = { listAutoSaved,       { " - Lists all auto saved Networks." } };
    mainMenu.commands["deleteSaved"] = { deleteSaved,         { " [path] - Deletes a saved Network." } };
    mainMenu.commands["export"]      = { exp,                 { " [fromPath] [toPath] - Copies a network to another directory(Downloads by default)." } };

    Menu* currentMenu = &mainMenu;

    while (true) {
        currentMenu->getInput();
    }

    NetworkBackProp TwoD_PlaneAI( vector<size_t>{ 28*28, 20, 20, 10 } );
    //NetworkBackProp TwoD_PlaneAI(std::string("19-0.000171"));

    train(TwoD_PlaneAI, 100, 100, 5000, 5000);

    //vector<double> result = testNetworkLearningSpeed(10, 2000, 0.1, TwoD_PlaneAI);

    std::cin.get();
}