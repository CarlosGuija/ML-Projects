#include "llm_project/data/character_vocabulary.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void fits_sorted_unique_characters() {
    llm::data::CharacterVocabulary vocabulary;
    vocabulary.fit("banana");

    assert(vocabulary.size() == 3);
    assert(vocabulary.token_to_char(0) == 'a');
    assert(vocabulary.token_to_char(1) == 'b');
    assert(vocabulary.token_to_char(2) == 'n');
}

void encodes_and_decodes_text() {
    llm::data::CharacterVocabulary vocabulary;
    vocabulary.fit("hello");

    const auto token_ids = vocabulary.encode("hello");
    assert(vocabulary.decode(token_ids) == "hello");
}

void rejects_unknown_characters() {
    llm::data::CharacterVocabulary vocabulary;
    vocabulary.fit("abc");

    bool threw = false;
    try {
        (void)vocabulary.encode("abcd");
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

}  // namespace

void run_character_vocabulary_tests() {
    fits_sorted_unique_characters();
    encodes_and_decodes_text();
    rejects_unknown_characters();
}
