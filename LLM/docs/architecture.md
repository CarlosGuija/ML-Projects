# Architecture

This project is a local C++ chat application built around `llama.cpp`.

## Current shape

- `llm chat`: interactive local chat backed by `llama-cli --conversation`.
- `llm serve-llama`: persistent local `llama-server` for RAG and web UI work.
- `llm generate-pretrained`: one-shot generation with a GGUF model.
- `llm "<prompt>"`: shorthand one-shot generation with the configured GGUF
  model.

Model files stay outside Git. The repository owns code, tests, docs, and small
examples only.

## Planned RAG flow

1. Ingest documents from `data/raw`.
2. Chunk documents into small passages.
3. Create embeddings for each chunk.
4. Store vectors and metadata locally.
5. Retrieve the most relevant chunks for a user question.
6. Add retrieved context to the chat prompt.
7. Generate the answer with `llama.cpp`.

The first RAG hook is `llm::rag::build_context_block`, which formats retrieved
chunks so the chat layer can inject them into the prompt.

## Why this split

`llama.cpp` handles model loading and inference. This project focuses on the
application layer around it: command parsing, local chat workflow, retrieval,
prompts, tests, and eventually a small local web UI. For interactive and web
work, `llama-server` is the preferred backend because it keeps the model loaded
between requests.
