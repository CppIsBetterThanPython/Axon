#pragma once

#include "pch.h"

namespace axon {

    // Abstract interface for passforward network interfaces
    class NetworkBase {
        std::vector<size_t> structure;
        size_t netSize;
    public:

        NetworkBase(const std::vector<size_t>& structure) : structure(structure) {
            netSize = structure.size();
        }

        virtual ~NetworkBase() = default;

        virtual void input(const std::vector<std::vector<double>>&) = 0;
        virtual void input(const std::vector<double>& input) = 0;
        virtual void calculate() = 0;
        [[nodiscard]] virtual std::vector<double> getAnswerVector() const = 0;
        [[nodiscard]] virtual std::vector<std::vector<double>> getAnswerVectors() const = 0;

        // Gets strongest node
        virtual size_t getAnswer() const {
            std::vector<double> answer = getAnswerVector();
            return getLargestID(answer);
        }

        // Gets if the network correctly guessed
        virtual bool isAnswerCorrect(const std::vector<double>& expectedAnswers) { return (getLargestID(expectedAnswers) == getAnswer()); }

        inline size_t size()            const { return netSize; }
        inline size_t inputLayerSize()  const { return structure.front(); }
        inline size_t outputLayerSize() const { return structure.back(); }

        inline const std::vector<size_t>& getStructure()      const { return structure; }
        inline size_t getStructure(size_t index) const { return structure[index]; }
    };

}