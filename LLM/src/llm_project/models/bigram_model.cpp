#include "llm_project/models/bigram_model.hpp"

#include <numeric>
#include <stdexcept>

namespace llm::models {

BigramModel::BigramModel(llm::data::CharacterVocabulary vocabulary)
    : vocabulary_(std::move(vocabulary)),
      transitions_(vocabulary_.size(), std::vector<std::uint64_t>(vocabulary_.size(), 1)) {
    if (vocabulary_.size() == 0) {
        throw std::invalid_argument("BigramModel requires a non-empty vocabulary.");
    }
}

void BigramModel::train(const std::string& text) {
    const auto token_ids = vocabulary_.encode(text);
    if (token_ids.size() < 2) {
        return;
    }

    for (std::size_t index = 1; index < token_ids.size(); ++index) {
        ++transitions_[token_ids[index - 1]][token_ids[index]];
    }
}

std::string BigramModel::generate(
    const std::string& prompt,
    const std::size_t max_new_tokens,
    const std::uint32_t seed
) const {
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt must not be empty.");
    }

    auto token_ids = vocabulary_.encode(prompt);
    std::mt19937 rng(seed);

    for (std::size_t count = 0; count < max_new_tokens; ++count) {
        token_ids.push_back(sample_next(token_ids.back(), rng));
    }

    return vocabulary_.decode(token_ids);
}

const llm::data::CharacterVocabulary& BigramModel::vocabulary() const {
    return vocabulary_;
}

std::size_t BigramModel::vocabulary_size() const {
    return vocabulary_.size();
}

std::size_t BigramModel::sample_next(const std::size_t current_token, std::mt19937& rng) const {
    const auto& row = transitions_.at(current_token);
    std::discrete_distribution<std::size_t> distribution(row.begin(), row.end());

    return distribution(rng);
}

}  // namespace llm::models
