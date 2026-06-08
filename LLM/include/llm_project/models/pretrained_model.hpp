#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace llm::models {

struct PretrainedGenerationOptions {
    std::filesystem::path executable{"llama-cli"};
    std::filesystem::path model_path;
    std::string system_prompt;
    std::string prompt;
    std::size_t max_tokens{128};
    float temperature{0.8F};
};

class PretrainedModelRunner {
public:
    explicit PretrainedModelRunner(PretrainedGenerationOptions options);

    [[nodiscard]] std::vector<std::string> command_arguments() const;
    [[nodiscard]] std::string command_line() const;
    [[nodiscard]] std::string chat_command_line() const;
    [[nodiscard]] std::string server_command_line(std::uint16_t port = 8080) const;

    int generate() const;
    int chat() const;
    int serve(std::uint16_t port = 8080) const;
    [[nodiscard]] std::string generate_text() const;

private:
    PretrainedGenerationOptions options_;
};

[[nodiscard]] std::string quote_command_argument(const std::string& argument);

}  // namespace llm::models
