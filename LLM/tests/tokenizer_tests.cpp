#include "llm_project/inference/tokenizer.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

void tokenizes_words_separated_by_whitespace() {
    const llm::inference::Tokenizer tokenizer;
    const auto tokens = tokenizer.tokenize("  build   a local\tLLM\nin C++  ");

    const std::vector<std::string> expected{"build", "a", "local", "LLM", "in", "C++"};
    assert(tokens == expected);
}

void detokenizes_with_single_spaces() {
    const llm::inference::Tokenizer tokenizer;
    const std::vector<std::string> tokens{"small", "model", "runtime"};

    assert(tokenizer.detokenize(tokens) == "small model runtime");
}

void handles_empty_input() {
    const llm::inference::Tokenizer tokenizer;

    assert(tokenizer.tokenize("").empty());
    assert(tokenizer.detokenize({}).empty());
}

}  // namespace

int main() {
    extern void run_chat_session_tests();
    extern void run_character_vocabulary_tests();
    extern void run_bigram_model_tests();
    extern void run_pretrained_model_tests();
    extern void run_rag_retrieval_tests();

    tokenizes_words_separated_by_whitespace();
    detokenizes_with_single_spaces();
    handles_empty_input();
    run_chat_session_tests();
    run_character_vocabulary_tests();
    run_bigram_model_tests();
    run_pretrained_model_tests();
    run_rag_retrieval_tests();

    return 0;
}
