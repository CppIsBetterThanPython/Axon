#pragma once

namespace axon {

    struct TestData {
        std::vector<double> values = {};
        explicit TestData(const std::vector<double>& values) : values(values) {}
        explicit TestData(std::vector<double>&& values) : values(std::move(values)) {}

        double& operator[](size_t i) { return values[i]; }
        const double operator[](size_t i) const { return values[i]; }

        const size_t size() const { return values.size(); }
    };

    struct Answer {
        std::vector<double> values = {};
        explicit Answer(const std::vector<double>& values) : values(values) {}
        explicit Answer(std::vector<double>&& values) : values(std::move(values)) {}

        double& operator[](size_t i) { return values[i]; }
        const double operator[](size_t i) const { return values[i]; }

        const size_t size() const { return values.size(); }
    };

    struct Test {
        TestData input;
        Answer expected;

        explicit Test(const TestData& data, const Answer& answer) : input(data), expected(answer) {}
        explicit Test(TestData&& data, Answer&& answer) : input(std::move(data)), expected(std::move(answer)) {}
    };

    struct TestResult {
        double cost;
        double accuracy;
    };

    class NetworkBackPropBase {

        virtual TestResult TestSet(const std::vector<Test>& testSet) = 0;
        virtual TestResult TrainSet(const std::vector<Test>& testSet, double learningRate) = 0;
    };

}