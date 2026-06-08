#include "llm_project/data/character_vocabulary.hpp"
#include "llm_project/models/bigram_model.hpp"

#include <cassert>
#include <stdexcept>

namespace {

void trains_and_generates_more_text() {
    llm::data::CharacterVocabulary vocabulary;
    vocabulary.fit("abababab");

    llm::models::BigramModel model(vocabulary);
    model.train("abababab");

    const auto output = model.generate("a", 8, 7);
    assert(output.size() == 9);
    assert(output.front() == 'a');
}

void rejects_empty_prompt() {
    llm::data::CharacterVocabulary vocabulary;
    vocabulary.fit("ab");

    llm::models::BigramModel model(vocabulary);

    bool threw = false;
    try {
        (void)model.generate("", 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

}  // namespace

void run_bigram_model_tests() {
    trains_and_generates_more_text();
    rejects_empty_prompt();
}
