#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#include "Mnist.h"

using nbp = NetworkBackProp;

uint32_t swap_bytes(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |      // Move byte 0 to byte 3
        ((value << 8) & 0x00FF0000) |      // Move byte 1 to byte 2
        ((value >> 8) & 0x0000FF00) |      // Move byte 2 to byte 1
        ((value << 24) & 0xFF000000);       // Move byte 3 to byte 0
}

uint32_t read_uint32(std::ifstream& file) {
    uint32_t value;

    file.read(reinterpret_cast<char*>(&value), 4);

    return swap_bytes(value);
}

uint8_t read_uint8(std::ifstream& file) {
    uint8_t value;

    file.read(reinterpret_cast<char*>(&value), 1);

    return value;
}

std::vector<Test> getImageSet(const std::string& image_path, const std::string& lable_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(lable_path, std::ios::binary);

    if (!image_file || !label_file) {
        std::cerr << "Error opening file!" << std::endl;
        return {};
    }

    // Read the image file header
    uint32_t magic_number = read_uint32(image_file);
    uint32_t num_images = read_uint32(image_file);
    uint32_t num_rows = read_uint32(image_file);
    uint32_t num_cols = read_uint32(image_file);

    std::cout << "Magic Number (Images): " << magic_number << std::endl;
    std::cout << "Number of Images: " << num_images << std::endl;
    std::cout << "Image Dimensions: " << num_rows << "x" << num_cols << std::endl;

    // Read the label file header
    magic_number = read_uint32(label_file);
    uint32_t num_labels = read_uint32(label_file);

    std::cout << "Magic Number (Labels): " << magic_number << std::endl;
    std::cout << "Number of Labels: " << num_labels << std::endl;

    std::vector<Test> testSet;

    testSet.reserve(num_images);

    // Read the image data and labels
    for (uint32_t i = 0; i < num_images; ++i) {
        TestData image(std::vector<double>(num_cols * num_rows));

        // Read image pixels
        for (uint32_t j = 0; j < num_cols * num_rows; j++) {
            image[j] = read_uint8(image_file);  // 28x28 pixel values
        }

        // Read label
        uint8_t label = read_uint8(label_file);

        // You can further process the image here, e.g., storing in a structure or array

        Answer answer(std::vector<double>(10, 0));

        answer[label] = 1;

        Test test = Test{ image, answer };
        testSet.push_back(test);
    }

    image_file.close();
    label_file.close();

    return testSet;
}

std::vector<Test> getTrainSet() {
    return getImageSet(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte"
    );
}

std::vector<Test> getTestSet() {
    return getImageSet(
        "data/t10k-images.idx3-ubyte",
        "data/t10k-labels.idx1-ubyte"
    );
}