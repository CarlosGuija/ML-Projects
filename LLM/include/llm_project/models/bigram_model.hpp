#pragma once

#include "llm_project/data/character_vocabulary.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace llm::models {

class BigramModel {
public:
    explicit BigramModel(llm::data::CharacterVocabulary vocabulary);

    void train(const std::string& text);

    [[nodiscard]] std::string generate(
        const std::string& prompt,
        std::size_t max_new_tokens,
        std::uint32_t seed = 42
    ) const;

    [[nodiscard]] const llm::data::CharacterVocabulary& vocabulary() const;
    [[nodiscard]] std::size_t vocabulary_size() const;

private:
    [[nodiscard]] std::size_t sample_next(std::size_t current_token, std::mt19937& rng) const;

    llm::data::CharacterVocabulary vocabulary_;
    std::vector<std::vector<std::uint64_t>> transitions_;
};

}  // namespace llm::models
