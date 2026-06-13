# Architecture

This project is a local C++ multimodal chat application built around
`llama.cpp`.

## Current shape

- `llm chat`: interactive local chat backed by `llama-cli --conversation`.
- `llm web`: starts the local `llama-server` backend and opens the browser.
- Multimodal support is delegated to `llama.cpp`, the selected GGUF model, and
  an optional compatible `mmproj` projector.

Model files stay outside Git. The repository owns code, tests, docs, and small
examples only.

## Why this split

`llama.cpp` handles model loading and inference. This project focuses on the
application layer around it: command parsing, local chat workflow, multimodal
server startup, prompts, tests, and a small local web UI. For interactive and
web work, `llama-server` is the preferred backend because it keeps the model
loaded between requests.
