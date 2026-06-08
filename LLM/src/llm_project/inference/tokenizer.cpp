#include "llm_project/inference/tokenizer.hpp"

#include <cctype>
#include <sstream>

namespace llm::inference {

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (const unsigned char character : text) {
        if (std::isspace(character) != 0) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }

        current.push_back(static_cast<char>(character));
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

std::string Tokenizer::detokenize(const std::vector<std::string>& tokens) const {
    std::ostringstream output;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }

        output << tokens[index];
    }

    return output.str();
}

}  // namespace llm::inference
