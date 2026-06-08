#include "llm_project/rag/retrieval.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

void builds_context_block_from_retrieved_chunks() {
    const std::vector<llm::rag::RetrievedChunk> chunks{
        {.source = "docs/intro.md", .text = "RAG adds external context.", .score = 0.91F},
        {.source = "docs/cpp.md", .text = "The runtime is written in C++.", .score = 0.83F},
    };

    const auto context = llm::rag::build_context_block(chunks);

    assert(context.find("Relevant context:") != std::string::npos);
    assert(context.find("[1] docs/intro.md") != std::string::npos);
    assert(context.find("RAG adds external context.") != std::string::npos);
    assert(context.find("[2] docs/cpp.md") != std::string::npos);
}

void empty_chunks_build_empty_context() {
    assert(llm::rag::build_context_block({}).empty());
}

}  // namespace

void run_rag_retrieval_tests() {
    builds_context_block_from_retrieved_chunks();
    empty_chunks_build_empty_context();
}
