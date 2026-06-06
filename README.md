# Axon

Axon is a cross-platform, open source machine learning library written in C++.
Currently it only supports the inference and training of fully connected neural networks with plans for implementing things like CNNs, Transformers, and reinforcement learning agents later on. The only API is also in `C++` with possible plans to implement `js` or `python`APIs.

It has both a CPU and GPU backend to allow for more flexibility. There are plans to implement other backends such as multithreaded and `WASM` in future.

The GPU backend is implemented in OpenCL, which is a cross platform library for writing GPU compute kernels.

## Intall

### Prerequisites

#### CMake

This is the build system used for the project. It is required to compile the library.

To install CMake, run `sudo apt install CMake` on Ubuntu and `winget install kitware.cmake` on Windows.

Please note you must also have a `C++` compiler (such as `g++`) installed to compile the code. This will also be required to write code using the library.

#### OpenCL

This is required for the GPU backend. 

To install, run `sudo apt install opencl-headers ocl-icd-opencl-dev` on Ubuntu and `vcpkg install opencl` on Windows.

### Compilation

TEMPORARILY EXCLUDED

## Usage

The API has been designed to be similar to that of other Neural Network libraries.

A simple example goes as follows.

```c++
#include "Axon/Network.cpp"

using axon;

int main() {
    // This defines a network of a given structure and asks the library to use the CPU interface.
    std::unique_ptr<Network> net = Network::createNetwork({3, 5, 2}, Network::Interface::CPU);

    // The input into the network.
    std::vector<double> input = {0.2, -0.4, 1.0};

    // Inputs the vector and peforms a forward pass
    net->input(input);
    net->calculate();

    auto output = net->getAnswerVector();
    size_t prediction = net->getAnswer();

    std::cout << "Output: ";
    for (double activation : output) std::cout << activation << " ";

    std::cout << "\n";
    std::cout << "Prediction: " << prediction << std::endl;
}
```
