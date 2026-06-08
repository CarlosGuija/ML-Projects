#pragma once

#include <string>
#include <vector>

namespace llm::inference {

class Tokenizer {
public:
    [[nodiscard]] std::vector<std::string> tokenize(const std::string& text) const;
    [[nodiscard]] std::string detokenize(const std::vector<std::string>& tokens) const;
};

}  // namespace llm::inference
