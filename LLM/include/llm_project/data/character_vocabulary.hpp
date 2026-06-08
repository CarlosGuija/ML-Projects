#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm::data {

class CharacterVocabulary {
public:
    void fit(const std::string& text);

    [[nodiscard]] std::vector<std::size_t> encode(const std::string& text) const;
    [[nodiscard]] std::string decode(const std::vector<std::size_t>& token_ids) const;

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] char token_to_char(std::size_t token_id) const;
    [[nodiscard]] std::size_t char_to_token(char character) const;

private:
    std::vector<char> id_to_char_;
    std::unordered_map<char, std::size_t> char_to_id_;
};

}  // namespace llm::data
