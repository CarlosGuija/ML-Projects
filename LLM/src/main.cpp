#include "llm_project/models/pretrained_model.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  llm chat --model path/to/model.gguf "
        << "[--prompt \"hello\"] [--max-tokens 256] [--temperature 0.8] [--exe path/to/llama-cli]\n"
        << "  llm web --model path/to/model.gguf [--mmproj path/to/mmproj.gguf] "
        << "[--port 8080] [--exe path/to/llama-cli]\n"
        << "\n"
        << "Environment:\n"
        << "  LLAMA_CPP_CLI    path to llama-cli.exe\n"
        << "  LOCAL_LLM_MODEL  path to a local GGUF model\n"
        << "  LOCAL_LLM_MMPROJ path to a local multimodal projector GGUF\n";
}

std::string default_system_prompt() {
    return "You are a helpful local assistant. Answer in clear, concise English. "
           "Do not output thinking, analysis, chain-of-thought, or hidden reasoning. "
           "Return only the final answer.";
}

std::optional<std::string> argument_value(const int argc, char* argv[], const std::string& name) {
    for (int index = 0; index < argc - 1; ++index) {
        if (argv[index] == name) {
            return argv[index + 1];
        }
    }

    return std::nullopt;
}

std::optional<std::string> default_model_path() {
    const std::filesystem::path lm_studio_model{
        "C:\\Users\\cagui\\.lmstudio\\models\\lmstudio-community\\gemma-4-E2B-it-GGUF\\gemma-4-E2B-it-Q4_K_M.gguf"
    };

    if (std::filesystem::exists(lm_studio_model)) {
        return lm_studio_model.string();
    }

    return std::nullopt;
}

std::optional<std::string> default_mmproj_path() {
    const std::filesystem::path lm_studio_mmproj{
        "C:\\Users\\cagui\\.lmstudio\\models\\lmstudio-community\\gemma-4-E2B-it-GGUF\\mmproj-gemma-4-E2B-it-BF16.gguf"
    };

    if (std::filesystem::exists(lm_studio_mmproj)) {
        return lm_studio_mmproj.string();
    }

    return std::nullopt;
}

std::filesystem::path default_executable_path() {
    const std::filesystem::path local_llama_cli{"tools\\llama.cpp\\llama-cli.exe"};
    if (std::filesystem::exists(local_llama_cli)) {
        return local_llama_cli;
    }

    return "llama-cli";
}

void open_browser_after_server_start(const std::string& port) {
#ifdef _WIN32
    const std::string url = "http://127.0.0.1:" + port;
    const std::string command = "cmd /c start \"\" \"" + url + "\"";
    (void)std::system(command.c_str());
#else
    (void)port;
#endif
}

llm::models::PretrainedGenerationOptions pretrained_options_from_args(
    const int argc,
    char* argv[],
    const std::string& prompt,
    const std::size_t default_max_tokens
) {
    auto model_path = argument_value(argc, argv, "--model");
    if (!model_path.has_value()) {
        if (const auto* env_model = std::getenv("LOCAL_LLM_MODEL"); env_model != nullptr) {
            model_path = env_model;
        }
    }
    if (!model_path.has_value()) {
        model_path = default_model_path();
    }

    if (!model_path.has_value()) {
        throw std::invalid_argument("Missing required --model argument or LOCAL_LLM_MODEL environment variable.");
    }

    llm::models::PretrainedGenerationOptions options;
    options.executable = default_executable_path();
    options.model_path = *model_path;
    if (const auto mmproj_path = argument_value(argc, argv, "--mmproj"); mmproj_path.has_value()) {
        options.mmproj_path = *mmproj_path;
    } else if (const auto* env_mmproj = std::getenv("LOCAL_LLM_MMPROJ"); env_mmproj != nullptr) {
        options.mmproj_path = env_mmproj;
    } else if (const auto default_mmproj = default_mmproj_path(); default_mmproj.has_value()) {
        options.mmproj_path = *default_mmproj;
    }
    options.prompt = prompt;
    options.max_tokens = default_max_tokens;
    options.system_prompt =
        argument_value(argc, argv, "--system").value_or(
            "You are a helpful local assistant. Answer in clear, concise English. "
            "Do not output thinking, analysis, chain-of-thought, or hidden reasoning. "
            "Return only the final answer."
        );

    if (const auto executable = argument_value(argc, argv, "--exe"); executable.has_value()) {
        options.executable = *executable;
    } else if (const auto* env_executable = std::getenv("LLAMA_CPP_CLI"); env_executable != nullptr) {
        options.executable = env_executable;
    }

    if (const auto max_tokens = argument_value(argc, argv, "--max-tokens"); max_tokens.has_value()) {
        options.max_tokens = static_cast<std::size_t>(std::stoul(*max_tokens));
    }

    if (const auto temperature = argument_value(argc, argv, "--temperature"); temperature.has_value()) {
        options.temperature = std::stof(*temperature);
    }

    return options;
}

int run_chat(const int argc, char* argv[]) {
    const auto has_model_argument = argument_value(argc, argv, "--model").has_value();
    const auto has_model_environment = std::getenv("LOCAL_LLM_MODEL") != nullptr;
    const auto has_default_model = default_model_path().has_value();
    if (!has_model_argument && !has_model_environment && !has_default_model) {
        print_usage();
        return 2;
    }

    const auto prompt = argument_value(argc, argv, "--prompt").value_or("Hello");
    auto options = pretrained_options_from_args(argc, argv, prompt, 256);
    options.system_prompt = default_system_prompt();

    const llm::models::PretrainedModelRunner runner(options);
    return runner.chat();
}

int run_web_server(const int argc, char* argv[]) {
    auto options = pretrained_options_from_args(argc, argv, "server", 128);
    const auto port = argument_value(argc, argv, "--port").value_or("8080");

    const llm::models::PretrainedModelRunner runner(options);
    return runner.serve(static_cast<std::uint16_t>(std::stoul(port)));
}

int run_web(const int argc, char* argv[]) {
    const auto port = argument_value(argc, argv, "--port").value_or("8080");
    open_browser_after_server_start(port);
    return run_web_server(argc, argv);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "chat") {
            return run_chat(argc - 2, argv + 2);
        }

        if (argc > 1 && std::string(argv[1]) == "web") {
            return run_web(argc - 2, argv + 2);
        }

        print_usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
