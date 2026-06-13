#include "llm_project/models/pretrained_model.hpp"

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

std::vector<std::string> chat_arguments(const PretrainedGenerationOptions& options) {
    std::vector<std::string> arguments{
        to_string(options.executable),
        "--model",
        to_string(options.model_path),
    };

    if (!options.system_prompt.empty()) {
        arguments.push_back("--system-prompt");
        arguments.push_back(options.system_prompt);
    }

    arguments.insert(arguments.end(), {
        "--prompt",
        options.prompt,
        "--n-predict",
        to_string(options.max_tokens),
        "--temp",
        to_string(options.temperature),
    });

    return arguments;
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

std::string PretrainedModelRunner::chat_command_line() const {
    const auto arguments = chat_arguments(options_);

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
