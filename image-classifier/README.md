# Clasificador de imagenes: perros vs gatos

Proyecto en C# con ML.NET para entrenar un clasificador de perros vs gatos usando transfer learning con una red ResNet preentrenada.

## Estado Actual

El proyecto usa .NET SDK 8 y ML.NET. La estructura esperada de datos ya existe en esta maquina:

```text
data/raw/train/cats  -> imagenes de gatos para entrenamiento
data/raw/train/dogs  -> imagenes de perros para entrenamiento
data/raw/test/cats   -> imagenes de gatos para evaluacion final
data/raw/test/dogs   -> imagenes de perros para evaluacion final
```

`data/raw/train` se divide internamente en entrenamiento y validacion. `data/raw/test` se usa al final para medir la accuracy real.

## Requisitos

- .NET SDK 8 o superior.
- Imagenes organizadas por carpeta de clase: `cats` y `dogs`.

Comprueba .NET:

```powershell
dotnet --info
```

Si PowerShell no reconoce `dotnet`, usa temporalmente:

```powershell
$env:Path += ';C:\Program Files\dotnet'
```

## Preparar

Desde la carpeta del repositorio:

```powershell
dotnet restore
dotnet build
```

## Entrenar

Entrenamiento recomendado:

```powershell
dotnet run -- train
```

Equivalente:

```powershell
dotnet run -- train-pretrained
```

El modelo se guarda en:

```text
models/dog-cat-pretrained.zip
```

Al terminar, el programa evalua automaticamente contra `data/raw/test`.

## Prueba Rapida

Antes de entrenar con todo el dataset, puedes probar el pipeline con pocas imagenes:

```powershell
dotnet run -- train --epochs 2 --max-images-per-class 100
```

Si eso termina bien, lanza el entrenamiento completo:

```powershell
dotnet run -- train --epochs 80 --batch-size 32
```

## Progreso

La primera ejecucion puede tardar antes de imprimir epochs porque ML.NET prepara/cachea caracteristicas del modelo preentrenado.

Mientras trabaja, veras mensajes como:

```text
Sigue entrenando... 18:30:00
```

Cuando ML.NET empiece a reportar metricas, veras:

```text
Epoch 1/80
Epoch 2/80
```

## Opciones Utiles

Entrenar mas epocas:

```powershell
dotnet run -- train --epochs 100
```

Usar una arquitectura mas pesada:

```powershell
dotnet run -- train --arch resnet101
```

Probar un learning rate mas bajo:

```powershell
dotnet run -- train --learning-rate 0.005
```

Ejemplo combinado:

```powershell
dotnet run -- train --epochs 100 --batch-size 32 --arch resnet101 --learning-rate 0.005
```

Arquitecturas disponibles:

- `resnet50`: default, buen equilibrio.
- `resnet101`: mas lento, puede mejorar accuracy.
- `mobilenet`: mas ligero.
- `inception`: alternativa para experimentar.

## Predecir

Con una imagen nueva:

```powershell
dotnet run -- predict --image ruta/a/la/imagen.jpg
```

Equivalente:

```powershell
dotnet run -- predict-pretrained --image ruta/a/la/imagen.jpg
```

Con una ruta de modelo especifica:

```powershell
dotnet run -- predict --image ruta/a/la/imagen.jpg --model-path models/dog-cat-pretrained.zip
```

## Imagenes Invalidas

Si aparece un aviso como:

```text
Aviso: se ignoraron X archivos con extension JPG/PNG pero contenido no compatible.
```

significa que algunos archivos tienen extension `.jpg`, `.jpeg` o `.png`, pero su contenido real es BMP o esta corrupto. El programa los ignora para evitar errores de decodificacion.

## Archivos Principales

- `ImageClassifier.csproj`: dependencias .NET y ML.NET.
- `src/Program.cs`: entrada del programa.
- `src/TrainPretrained.cs`: entrenamiento e inferencia con transfer learning.
- `src/Shared.cs`: opciones, carga de datos y utilidades.

## Salidas Generadas

Estas carpetas se crean durante el uso y estan ignoradas por Git:

```text
models/   -> modelos entrenados
outputs/  -> cache de ML.NET
bin/      -> compilacion .NET
obj/      -> archivos intermedios .NET
```
