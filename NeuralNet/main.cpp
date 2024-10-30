#ifndef SSIZE_T
//#include "hdf5.h"  // Include HDF5 first
#endif
#include <cmath> //for basic math functions, for exponent used in sigmoid
#include <cstdlib> //for random numbes
#include <ctime> //so is this
#include <iostream>
#include <deque>
#include <utility>
#include <algorithm>  // For std::sort

#include <chrono>


#include "NeuralNet.h"

using namespace std;

vector<vector<vector<double>>> generateTestSet(int size) {
    vector<vector<vector<double>>> testSet(size);

    for(int i = 0; i < size; i++) {
        vector<double> x = {static_cast<double>(rand() % 100)+RandomReal(), static_cast<double>(rand() % 100)+RandomReal()};
        vector<double> y = {static_cast<double>(rand() % 100)+RandomReal(), static_cast<double>(rand() % 100)+RandomReal()};

        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());

        vector<double> point = {static_cast<double>(rand() % 100)+RandomReal(), static_cast<double>(rand() % 100)+RandomReal()};

        vector<double> input = {x[0], y[0], x[1], y[1], point[0], point[1]};
        
        vector<double> answer(2);

        if (isInRange(point[0], x[0], x[1]) && isInRange(point[1], y[0], y[1])) {
            answer = {1, 0};
        }
        else {
            answer = {0, 1};
        }

        vector<vector<double>> test = {input, answer};

        testSet[i] = test;
    }

    return testSet;
}

//returns cost then accuracy
vector<double> constantLearningApproach(Network& network) {
    return network.improveNetworkBackPropogation(generateTestSet(1000), 1);
}

// smoothing factor closer to 1 values newer values more, 0 values older values more
double EWMA(double pastAverage, double currentCost, double smoothingFactor, int time) {
    double newAverage = smoothingFactor * pastAverage + (1 - smoothingFactor) * currentCost;

    newAverage /= 1 - pow(smoothingFactor, time);

    return newAverage;
};

void train(Network& network) {
    double currentCost = 1;
    double weightedCost = constantLearningApproach(network)[0];

    int instanceCounter = 0;

    for (;;) {
        vector<double> result = constantLearningApproach(network);
        currentCost = result[0];
        weightedCost = EWMA(weightedCost, currentCost, 0.8, instanceCounter + 1);

        std::cout << instanceCounter << ":" << endl;
        cout << "Accuracy: " << result[1] << endl;
        cout << "cost: " << weightedCost << endl;

        instanceCounter++;
    }
}

vector<double> testNetworkLearningSpeed(int testLength, int upperBound, double targetCost, Network& network) {
    int amountCompleted = testLength;

    double averageCount = 0.0;
    double averageElapsed = 0.0;

    for (int i = 0; i < testLength; i++) {
        double currentCost = 1;
        double weightedCost = constantLearningApproach(network)[0];

        int instanceCounter = 0;

        // Start time measurement
        auto start = std::chrono::high_resolution_clock::now();

        while (weightedCost > targetCost && instanceCounter <= upperBound) {
            std::cout << instanceCounter << ":" << endl;
            vector<double> result = constantLearningApproach(network);
            currentCost = result[0];
            weightedCost = EWMA(weightedCost, currentCost, 0.8, instanceCounter + 1);

            cout << "Accuracy: " << result[1] << endl;
            cout << "cost: " << weightedCost << endl;

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

    std::cout << "Amount completed: " << amountCompleted << endl;

    return { averageCount, averageElapsed };
}

int main() {
    srand(static_cast<int>(time(NULL)));
    vector<int> Structure{6, 5, 5, 2};
    Network TwoD_PlaneAI(Structure);

    train(TwoD_PlaneAI);

    vector<double> result = testNetworkLearningSpeed(10, 20000, 0.1, TwoD_PlaneAI);

    double averageElapsed = result[1];
    double averageCount = result[0];

    std::cout << "Average count: " << averageCount << endl;

    std::cout << "Average elapsed time: " << averageElapsed << " seconds\n";

    std::cin.get();

    return 0;
}