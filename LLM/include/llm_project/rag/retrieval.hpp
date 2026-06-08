#pragma once

#include <string>
#include <vector>

namespace llm::rag {

struct RetrievedChunk {
    std::string source;
    std::string text;
    float score{0.0F};
};

[[nodiscard]] std::string build_context_block(const std::vector<RetrievedChunk>& chunks);

}  // namespace llm::rag
