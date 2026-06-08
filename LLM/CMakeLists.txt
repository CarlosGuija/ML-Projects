cmake_minimum_required(VERSION 3.20)

project(LLM
    VERSION 0.1.0
    DESCRIPTION "A C++ foundation for local LLM experiments"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(llm_core
    src/llm_project/chat/chat_session.cpp
    src/llm_project/data/character_vocabulary.cpp
    src/llm_project/inference/tokenizer.cpp
    src/llm_project/models/bigram_model.cpp
    src/llm_project/models/pretrained_model.cpp
    src/llm_project/rag/retrieval.cpp
)

target_include_directories(llm_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(llm_core PUBLIC cxx_std_20)

add_executable(llm
    src/main.cpp
)

target_link_libraries(llm PRIVATE llm_core)

enable_testing()

add_executable(llm_tests
    tests/bigram_model_tests.cpp
    tests/chat_session_tests.cpp
    tests/character_vocabulary_tests.cpp
    tests/pretrained_model_tests.cpp
    tests/rag_retrieval_tests.cpp
    tests/tokenizer_tests.cpp
)

target_link_libraries(llm_tests PRIVATE llm_core)

add_test(NAME tokenizer_tests COMMAND llm_tests)
