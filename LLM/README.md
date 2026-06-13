# LLM

Aplicacion local en C++ para usar modelos LLM preentrenados con `llama.cpp`.
La meta del proyecto es tener una interfaz local multimodal para chat, texto,
imagenes y documentos mediante las capacidades de `llama.cpp` y del modelo
elegido.

Este proyecto no entrena LLMs, no hace fine-tuning y no modifica pesos. Carga
modelos GGUF externos y ejecuta inferencia local.

## Requisitos

- CMake 3.20 o superior.
- Compilador compatible con C++20.
- `llama.cpp` disponible localmente.
- Un modelo GGUF disponible en una ruta local.

Por defecto se busca:

```text
tools\llama.cpp\llama-cli.exe
```

Para `web`, `llama-server.exe` debe estar en la misma carpeta.

## Compilar

```powershell
cmake -S . -B build
cmake --build build
```

El ejecutable puede quedar en `.\build\llm.exe` o `.\build\Debug\llm.exe`,
segun el generador de CMake.

## Configurar Rutas

```powershell
$env:LLAMA_CPP_CLI=".\tools\llama.cpp\llama-cli.exe"
$env:LOCAL_LLM_MODEL="C:\ruta\al\modelo.gguf"
```

Opcional para modelos multimodales:

```powershell
$env:LOCAL_LLM_MMPROJ="C:\ruta\al\mmproj.gguf"
```

## Uso

Chat interactivo:

```powershell
.\build\llm.exe chat --max-tokens 256
```

Abrir servidor en navegador:

```powershell
.\build\llm.exe web --port 8080
```

Tambien puedes pasar rutas directamente:

```powershell
.\build\llm.exe chat --exe .\tools\llama.cpp\llama-cli.exe --model C:\ruta\al\modelo.gguf
```

Opciones principales: `--model`, `--prompt`, `--max-tokens`, `--temperature`,
`--system`, `--exe`, `--port`, `--mmproj`.

## Tests

```powershell
ctest --test-dir build --output-on-failure
```

## Estructura

- `src/`: implementacion C++.
- `include/`: cabeceras publicas.
- `tests/`: tests.
- `docs/`: notas tecnicas.
- `tools/`: herramientas locales como `llama.cpp`.
