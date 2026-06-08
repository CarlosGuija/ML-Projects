#include "llm_project/chat/chat_session.hpp"

#include <cassert>
#include <string>

namespace {

void builds_chat_prompt_with_history() {
    llm::chat::ChatSession session("Be brief.");
    session.add_user_message("Hello");
    session.add_assistant_message("Hello, how can I help?");
    session.add_user_message("What is RAG?");

    const auto prompt = session.build_prompt();

    assert(prompt.find("System: Be brief.") == std::string::npos);
    assert(prompt.find("User: Hello") != std::string::npos);
    assert(prompt.find("Assistant: Hello, how can I help?") != std::string::npos);
    assert(prompt.ends_with("Assistant:"));
}

void keeps_system_prompt_available_for_llama_cpp() {
    const llm::chat::ChatSession session("Answer in English.");

    assert(session.system_prompt() == "Answer in English.");
}

}  // namespace

void run_chat_session_tests() {
    builds_chat_prompt_with_history();
    keeps_system_prompt_available_for_llama_cpp();
}
