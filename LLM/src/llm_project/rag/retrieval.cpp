#include "llm_project/rag/retrieval.hpp"

#include <sstream>

namespace llm::rag {

std::string build_context_block(const std::vector<RetrievedChunk>& chunks) {
    if (chunks.empty()) {
        return "";
    }

    std::ostringstream context;
    context << "Relevant context:\n";

    for (std::size_t index = 0; index < chunks.size(); ++index) {
        context << "[" << index + 1 << "] " << chunks[index].source << "\n";
        context << chunks[index].text << "\n\n";
    }

    return context.str();
}

}  // namespace llm::rag
