#ifndef SSIZE_T
//#include "hdf5.h"  // Include HDF5 first
#endif
#include <cmath> //for basic math functions, for exponent used in sigmoid
#include <cstdlib> //for random numbes
#include <ctime> //so is this
#include <iostream>
#include <vector> // (._.) < Really? ... )
#include <deque>
#include <utility>
#include <tuple>//to return multiple different type values
#include <algorithm>  // For std::sort

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

double learningRate(int x, double constant) {
    return (1/constant)*exp(-x)+(constant-1)/constant;
}

int main() {
    srand(static_cast<int>(time(NULL)));
    vector<int> Structure{6, 5, 5, 2};
    Network TwoD_PlaneAI(Structure);
    /*for(int i = 0; i > -1; i++) {
        cout << i << ":" << endl;
        TwoD_PlaneAI.improveNetworkBackPropogation(generateTestSet(1000), 1.0001);
    }*/

    cout << "Phase One:" << endl;
    for(int i = 0; i < 10000; i++) {
        cout << i << ":" << endl;
        TwoD_PlaneAI.improveNetworkBackPropogation(generateTestSet(1000), 1.0001);
    }
    cout << "Phase Two:" << endl;
    for(int i = 10000; i < 30000; i++) {
        cout << i << ":" << endl;
        TwoD_PlaneAI.improveNetworkBackPropogation(generateTestSet(10000), 0.1);
    }
    cout << "Phase Three:" << endl;
    for(int i = 30000; i > -1; i++) {
        cout << i << ":" << endl;
        TwoD_PlaneAI.improveNetworkBackPropogation(generateTestSet(100000), 0.01);
    }

    return 0;
}