#include "llm_project/models/pretrained_model.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace llm::models {

namespace {

std::string to_string(const std::filesystem::path& path) {
    return path.string();
}

std::string to_string(const std::size_t value) {
    return std::to_string(value);
}

std::string to_string(const float value) {
    std::ostringstream output;
    output << value;
    return output.str();
}

std::filesystem::path sibling_executable(const std::filesystem::path& executable, const std::string& filename) {
    if (executable.has_parent_path()) {
        return executable.parent_path() / filename;
    }

    return filename;
}

}  // namespace

PretrainedModelRunner::PretrainedModelRunner(PretrainedGenerationOptions options)
    : options_(std::move(options)) {
    if (options_.model_path.empty()) {
        throw std::invalid_argument("A pretrained model path is required.");
    }

    if (options_.prompt.empty()) {
        throw std::invalid_argument("A prompt is required.");
    }

    if (options_.max_tokens == 0) {
        throw std::invalid_argument("max_tokens must be greater than zero.");
    }
}

std::vector<std::string> PretrainedModelRunner::command_arguments() const {
    std::vector<std::string> arguments{
        to_string(options_.executable),
        "--model",
        to_string(options_.model_path),
    };

    if (!options_.system_prompt.empty()) {
        arguments.push_back("--system-prompt");
        arguments.push_back(options_.system_prompt);
    }

    arguments.insert(arguments.end(), {
        "--prompt",
        options_.prompt,
        "--n-predict",
        to_string(options_.max_tokens),
        "--temp",
        to_string(options_.temperature),
        "--no-display-prompt",
        "--simple-io",
        "--reasoning",
        "off",
    });

    return arguments;
}

std::string PretrainedModelRunner::command_line() const {
    const auto arguments = command_arguments();
    std::ostringstream command;

    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) {
            command << ' ';
        }

        command << quote_command_argument(arguments[index]);
    }

    return command.str();
}

std::string PretrainedModelRunner::chat_command_line() const {
    auto arguments = command_arguments();
    arguments.push_back("--conversation");

    std::ostringstream command;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) {
            command << ' ';
        }

        command << quote_command_argument(arguments[index]);
    }

    return command.str();
}

std::string PretrainedModelRunner::server_command_line(const std::uint16_t port) const {
    const auto server_executable = sibling_executable(options_.executable, "llama-server.exe");
    std::vector<std::string> arguments{
        to_string(server_executable),
        "--model",
        to_string(options_.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        std::to_string(port),
        "--reasoning",
        "off",
    };

    if (!options_.mmproj_path.empty()) {
        arguments.push_back("--mmproj");
        arguments.push_back(to_string(options_.mmproj_path));
    }

    std::ostringstream command;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0) {
            command << ' ';
        }

        command << quote_command_argument(arguments[index]);
    }

    return command.str();
}

int PretrainedModelRunner::generate() const {
    if (!std::filesystem::exists(options_.model_path)) {
        throw std::runtime_error("Pretrained model file does not exist: " + to_string(options_.model_path));
    }

    return std::system(command_line().c_str());
}

int PretrainedModelRunner::chat() const {
    if (!std::filesystem::exists(options_.model_path)) {
        throw std::runtime_error("Pretrained model file does not exist: " + to_string(options_.model_path));
    }

    return std::system(chat_command_line().c_str());
}

int PretrainedModelRunner::serve(const std::uint16_t port) const {
    if (!std::filesystem::exists(options_.model_path)) {
        throw std::runtime_error("Pretrained model file does not exist: " + to_string(options_.model_path));
    }

    return std::system(server_command_line(port).c_str());
}

std::string PretrainedModelRunner::generate_text() const {
    if (!std::filesystem::exists(options_.model_path)) {
        throw std::runtime_error("Pretrained model file does not exist: " + to_string(options_.model_path));
    }

#ifdef _WIN32
    FILE* pipe = _popen(command_line().c_str(), "r");
#else
    FILE* pipe = popen(command_line().c_str(), "r");
#endif

    if (pipe == nullptr) {
        throw std::runtime_error("Failed to run pretrained model command.");
    }

    std::string output;
    std::array<char, 4096> buffer{};

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

#ifdef _WIN32
    const int result = _pclose(pipe);
#else
    const int result = pclose(pipe);
#endif

    if (result != 0) {
        throw std::runtime_error("Pretrained model command failed.");
    }

    return output;
}

std::string quote_command_argument(const std::string& argument) {
    if (argument.empty()) {
        return "\"\"";
    }

    const auto needs_quotes = argument.find_first_of(" \t\n\r\"") != std::string::npos;
    if (!needs_quotes) {
        return argument;
    }

    std::string quoted;
    quoted.reserve(argument.size() + 2);
    quoted.push_back('"');

    for (const char character : argument) {
        if (character == '"') {
            quoted.push_back('\\');
        }

        quoted.push_back(character);
    }

    quoted.push_back('"');
    return quoted;
}

}  // namespace llm::models
