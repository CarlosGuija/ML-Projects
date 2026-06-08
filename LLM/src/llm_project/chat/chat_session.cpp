#include "llm_project/chat/chat_session.hpp"

#include <sstream>
#include <utility>

namespace llm::chat {

namespace {

std::string role_label(const Role role) {
    switch (role) {
        case Role::System:
            return "System";
        case Role::User:
            return "User";
        case Role::Assistant:
            return "Assistant";
    }

    return "Unknown";
}

}  // namespace

ChatSession::ChatSession(std::string system_prompt) {
    system_prompt_ = std::move(system_prompt);
}

void ChatSession::add_user_message(std::string content) {
    messages_.push_back({Role::User, std::move(content)});
}

void ChatSession::add_assistant_message(std::string content) {
    messages_.push_back({Role::Assistant, std::move(content)});
}

std::string ChatSession::build_prompt() const {
    std::ostringstream prompt;

    for (const auto& message : messages_) {
        if (message.role == Role::System) {
            continue;
        }

        prompt << role_label(message.role) << ": " << message.content << "\n\n";
    }

    prompt << "Assistant:";
    return prompt.str();
}

const std::vector<Message>& ChatSession::messages() const {
    return messages_;
}

const std::string& ChatSession::system_prompt() const {
    return system_prompt_;
}

std::string ChatSession::default_system_prompt() {
    return "You are a helpful local assistant. Answer in clear, concise English. "
           "Do not output thinking, analysis, chain-of-thought, or hidden reasoning. "
           "Return only the final answer.";
}

}  // namespace llm::chat
