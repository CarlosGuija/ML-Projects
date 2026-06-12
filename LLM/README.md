# LLM

Local LLM Chat in C++ using `llama.cpp`, with a base prepared to grow into
Document Q&A / RAG and a local web UI. Large models and datasets stay outside
Git.

## Requisitos

- CMake 3.20 o superior
- Un compilador compatible con C++20

## Compilar

```powershell
cmake -S . -B build
cmake --build build
```

## Ejecutar

```powershell
.\build\Debug\llm.exe "hello local model"
```

En generadores de un solo tipo de build, el ejecutable puede quedar en:

```powershell
.\build\llm.exe "hello local model"
```

El ejecutable usa el modelo GGUF configurado y genera texto a partir del
prompt.

## Local Chat With llama.cpp

La ruta recomendada para modelos reales es usar pesos externos en formato GGUF
con `llama.cpp`. El repositorio no guarda esos pesos: `models/`, `data/` y los
formatos grandes quedan ignorados por Git.

Compila o descarga `llama.cpp` fuera de este repositorio, deja disponible el
binario `llama-cli`, y ejecuta un chat interactivo:

```powershell
.\build\llm.exe chat --exe .\tools\llama.cpp\llama-cli.exe --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf
```

Si defines `LLAMA_CPP_CLI`, no necesitas pasar `--exe` cada vez:

```powershell
$env:LLAMA_CPP_CLI=".\tools\llama.cpp\llama-cli.exe"
.\build\llm.exe chat --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf
```

`llm chat` launches `llama-cli` in persistent conversation mode, so the model is
loaded once instead of being restarted for every message.

Si ya tienes un modelo en LM Studio, puedes reutilizarlo sin copiarlo al
proyecto. Por ejemplo:

```powershell
$env:LLAMA_CPP_CLI=".\tools\llama.cpp\llama-cli.exe"
$env:LOCAL_LLM_MODEL="C:\Users\cagui\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf"
.\build\llm.exe chat --max-tokens 128
```

You can also pass that path directly with `--model`.

## Local llama.cpp Server

For RAG and the future local web UI, prefer a persistent HTTP backend:

```powershell
.\build\llm.exe serve-llama --port 8080
```

This starts `llama-server` on:

```text
http://127.0.0.1:8080
```

Keep that terminal open while another process, RAG tool, or web UI talks to the
server.

To start the server and open the browser in one command:

```powershell
.\build\llm.exe web
```

Or choose a port:

```powershell
.\build\llm.exe web --port 8080
```

This opens:

```text
http://127.0.0.1:8080
```

You can also use:

```powershell
.\build\llm.exe serve-llama --port 8080 --open
```

## Image Input

The local web UI can support image upload when the model has a multimodal
projector. This project autodetects the Gemma projector installed by LM Studio:

```powershell
C:\Users\cagui\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\mmproj-gemma-4-E2B-it-BF16.gguf
```

So the usual command is enough:

```powershell
.\build\llm.exe web
```

If you want to pass a projector explicitly:

```powershell
.\build\llm.exe web --mmproj C:\path\to\mmproj.gguf
```

or:

```powershell
.\build\llm.exe serve-llama --port 8080 --open --mmproj C:\path\to\mmproj.gguf
```

This does not copy model files into the repository. The web UI reads the model
and projector from their local paths.

## Probar llama.cpp

Primero comprueba que `llama-cli` funciona por separado:

```powershell
.\tools\llama.cpp\llama-cli.exe --help
```

Luego prueba un modelo GGUF directamente con `llama.cpp`:

```powershell
.\tools\llama.cpp\llama-cli.exe --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf --system-prompt "You are a helpful assistant. Answer in clear English." --prompt "What is an LLM? Answer in one sentence." --n-predict 80 --no-display-prompt --simple-io
```

Si esa prueba responde texto, conecta el mismo binario a este proyecto:

```powershell
$env:LLAMA_CPP_CLI=".\tools\llama.cpp\llama-cli.exe"
.\build\llm.exe generate-pretrained --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt "What is an LLM? Answer in one sentence." --max-tokens 80
```

Con `LOCAL_LLM_MODEL` definido:

```powershell
.\build\llm.exe generate-pretrained --prompt "What is an LLM? Answer in one sentence." --max-tokens 80
```

Y finalmente prueba el chat:

```powershell
.\build\llm.exe chat --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf --max-tokens 128
```

Tambien puedes usar un modelo de Hugging Face directamente desde `llama-cli`
con `-hf`, pero este proyecto usa por ahora rutas locales `--model` para evitar
descargas implicitas y mantener claro donde estan los pesos.

## One-Shot Generation

Tambien puedes ejecutar una sola generacion:

```powershell
.\build\llm.exe generate-pretrained --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt "Explain C++ in one sentence." --max-tokens 80
```

Tambien puedes indicar el binario explicitamente:

```powershell
.\build\llm.exe generate-pretrained --exe .\tools\llama.cpp\llama-cli.exe --model .\models\SmolLM2-135M-Instruct-Q4_K_M.gguf --prompt "Hello"
```

## Camino hacia RAG

La siguiente mejora sera RAG:

1. Cargar documentos desde `data/raw`.
2. Dividirlos en chunks.
3. Crear embeddings.
4. Guardar un indice vectorial local.
5. Recuperar contexto relevante por pregunta.
6. Inyectar ese contexto en el prompt del chat.

La base inicial esta en `include/llm_project/rag`.

## Tests

```powershell
ctest --test-dir build --output-on-failure
```

## Estructura

- `include/`: cabeceras publicas del proyecto.
- `src/`: implementacion C++.
- `tests/`: tests nativos registrados con CTest.
- `data/`: datasets locales ignorados por Git.
- `models/`: pesos/modelos externos ignorados por Git.
- `outputs/`: generaciones y resultados temporales ignorados por Git.
- `docs/`: notas tecnicas y documentacion del proyecto.

## Peso del proyecto

El codigo fuente de un proyecto C++ suele pesar poco. El tamano en GB aparecera
cuando agregues datasets, checkpoints, embeddings, cuantizaciones y salidas de
entrenamiento. Para archivos grandes conviene usar Git LFS o mantenerlos fuera
del repositorio principal y documentar como descargarlos.
