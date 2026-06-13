#include "llm_project/models/pretrained_model.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

namespace {

void rejects_missing_model_path() {
    try {
        (void)llm::models::PretrainedModelRunner({
            .model_path = "",
            .mmproj_path = "",
            .system_prompt = "Answer in English.",
            .prompt = "hello",
        });
    } catch (const std::invalid_argument&) {
        return;
    }

    assert(false);
}

void quotes_arguments_with_double_quotes() {
    const auto quoted = llm::models::quote_command_argument("say \"hola\"");

    assert(quoted == "\"say \\\"hola\\\"\"");
}

void builds_persistent_chat_command_line() {
    const llm::models::PretrainedModelRunner runner({
        .executable = "tools/llama.cpp/llama-cli.exe",
        .model_path = "models/model.gguf",
        .mmproj_path = "",
        .system_prompt = "Answer in English.",
        .prompt = "Hello",
        .max_tokens = 128,
        .temperature = 0.2F,
    });

    const auto command = runner.chat_command_line();

    assert(command.find("tools/llama.cpp/llama-cli.exe") != std::string::npos);
    assert(command.find("--model models/model.gguf") != std::string::npos ||
           command.find("--model \"models/model.gguf\"") != std::string::npos);
    assert(command.find("--system-prompt \"Answer in English.\"") != std::string::npos);
    assert(command.find("--prompt Hello") != std::string::npos ||
           command.find("--prompt \"Hello\"") != std::string::npos);
    assert(command.find("--n-predict 128") != std::string::npos);
    assert(command.find("--temp 0.2") != std::string::npos);
    assert(command.find("--system-prompt") != std::string::npos);
    assert(command.find("--prompt Hello") != std::string::npos ||
           command.find("--prompt \"Hello\"") != std::string::npos);
    assert(command.find("--conversation") == std::string::npos);
    assert(command.find("--simple-io") == std::string::npos);
    assert(command.find("--reasoning") == std::string::npos);
}

void builds_llama_server_command_line() {
    const llm::models::PretrainedModelRunner runner({
        .executable = "tools/llama.cpp/llama-cli.exe",
        .model_path = "models/model.gguf",
        .mmproj_path = "models/mmproj.gguf",
        .system_prompt = "Answer in English.",
        .prompt = "server",
    });

    const auto command = runner.server_command_line(9090);

    assert(command.find("tools/llama.cpp\\llama-server.exe") != std::string::npos ||
           command.find("tools/llama.cpp/llama-server.exe") != std::string::npos);
    assert(command.find("--host 127.0.0.1") != std::string::npos);
    assert(command.find("--port 9090") != std::string::npos);
    assert(command.find("--reasoning") == std::string::npos);
    assert(command.find("--mmproj models/mmproj.gguf") != std::string::npos ||
           command.find("--mmproj \"models/mmproj.gguf\"") != std::string::npos);
    assert(command.find("--system-prompt") == std::string::npos);
}

}  // namespace

void run_pretrained_model_tests() {
    rejects_missing_model_path();
    quotes_arguments_with_double_quotes();
    builds_persistent_chat_command_line();
    builds_llama_server_command_line();
}
