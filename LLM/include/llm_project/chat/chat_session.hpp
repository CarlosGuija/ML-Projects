#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llm::chat {

enum class Role {
    System,
    User,
    Assistant,
};

struct Message {
    Role role;
    std::string content;
};

class ChatSession {
public:
    explicit ChatSession(std::string system_prompt = default_system_prompt());

    void add_user_message(std::string content);
    void add_assistant_message(std::string content);

    [[nodiscard]] std::string build_prompt() const;
    [[nodiscard]] const std::string& system_prompt() const;
    [[nodiscard]] const std::vector<Message>& messages() const;

    static std::string default_system_prompt();

private:
    std::string system_prompt_;
    std::vector<Message> messages_;
};

}  // namespace llm::chat
