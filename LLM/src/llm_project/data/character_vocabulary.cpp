#include "llm_project/data/character_vocabulary.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace llm::data {

void CharacterVocabulary::fit(const std::string& text) {
    std::unordered_set<char> unique_characters(text.begin(), text.end());

    id_to_char_.assign(unique_characters.begin(), unique_characters.end());
    std::sort(id_to_char_.begin(), id_to_char_.end());

    char_to_id_.clear();
    for (std::size_t index = 0; index < id_to_char_.size(); ++index) {
        char_to_id_[id_to_char_[index]] = index;
    }
}

std::vector<std::size_t> CharacterVocabulary::encode(const std::string& text) const {
    std::vector<std::size_t> token_ids;
    token_ids.reserve(text.size());

    for (const char character : text) {
        token_ids.push_back(char_to_token(character));
    }

    return token_ids;
}

std::string CharacterVocabulary::decode(const std::vector<std::size_t>& token_ids) const {
    std::string text;
    text.reserve(token_ids.size());

    for (const std::size_t token_id : token_ids) {
        text.push_back(token_to_char(token_id));
    }

    return text;
}

std::size_t CharacterVocabulary::size() const {
    return id_to_char_.size();
}

char CharacterVocabulary::token_to_char(const std::size_t token_id) const {
    if (token_id >= id_to_char_.size()) {
        throw std::out_of_range("Token id is outside the vocabulary.");
    }

    return id_to_char_[token_id];
}

std::size_t CharacterVocabulary::char_to_token(const char character) const {
    const auto found = char_to_id_.find(character);
    if (found == char_to_id_.end()) {
        throw std::invalid_argument("Character is not in the vocabulary.");
    }

    return found->second;
}

}  // namespace llm::data
