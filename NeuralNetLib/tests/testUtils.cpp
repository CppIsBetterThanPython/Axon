/*

#include <gtest/gtest.h>

#include "utils.h"

constexpr double epsilon = 1e-16

TEST(SigmoidTest, Zero check) {
	double input = 0.0;

	double expected = 0.5;

	double output = Sigmoid(input);

	EXPECT_NEAR(output, expected, epsilon);
}

TEST(FlattenVectorTests, Normal) {
	std::vector<std::vector<int>> nested = {
		{0, 1, 5},
		{3},
		{2, 7}
	};

	std::vector<int> expected = { 0, 1, 5, 3, 2, 7 };

	std::vector<int> output = flattenVector(nested);

	EXPECT_EQ(output, expected)
}

TEST(FlattenVectorTests, Array) {
	std::vector<std::array<double>> nested = {
		{0.5, 0.6, 0.7},
		{0.1},
		{0.2, 0.2}
	};

	std::vector<double> expected = { 0.5, 0.6, 0.7 , 0.1, 0.2, 0.2 };

	std::vector<double> output = fattenVector(nested);

	EXPECT_EQ(output, expected);
}

TEST(GetLargestIDTests, Normal) {
	std::vector<int> vec = { 1, 2, 5, 9, 8 };

	size_t expected = 3;

	size_t output = getLargestID(vec);

	EXPECT_EQ(output, expected);
}

TEST(GetLargestIDTests, Chars) {
	std::vector<char> vec = { 'a', 'c', 'b', 's', 'z'};

	size_t expected = 3;

	size_t output = getLargestID(vec);

	EXPECT_EQ(output, expected);
}
*/