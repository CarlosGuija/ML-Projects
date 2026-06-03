# Image Classifier

Clasificador de perros vs gatos en C# con ML.NET, transfer learning y una app web ASP.NET Core para probar imagenes desde el navegador.

## Requisitos

- .NET SDK 8.
- Imagenes JPG o PNG organizadas por clase.

## Datos

El dataset debe estar organizado asi:

```text
data/raw/
  train/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

`train` se usa para entrenar y validar. `test` se usa para medir el resultado final.

## Preparar

```powershell
dotnet restore
dotnet build
```

## Entrenar

```powershell
dotnet run -- train
```

Prueba rapida con pocas imagenes:

```powershell
dotnet run -- train --epochs 2 --max-images-per-class 100
```

El modelo se guarda en:

```text
models/dog-cat-pretrained.zip
```

Al terminar el entrenamiento, si existe `data/raw/test`, el programa evalua automaticamente el modelo.

## Evaluar

Para evaluar un modelo ya entrenado sin volver a entrenar:

```powershell
dotnet run -- test
```

El test muestra una muestra de predicciones, accuracy, correctas/total y matriz de predicciones.

## Predecir Una Imagen

```powershell
dotnet run -- predict --image ruta/a/imagen.jpg
```

El comando imprime la prediccion, la confianza y las probabilidades para `gato` y `perro`.

## App Web

Levanta una interfaz web en C# para subir una foto y ver la prediccion con probabilidades:

```powershell
dotnet run -- web
```

La app abre el navegador automaticamente. Si necesitas abrirla manualmente, usa la URL que aparece en consola:

```text
http://localhost:5000
```

Si el puerto `5000` esta ocupado, la app usa el siguiente puerto libre y lo muestra en consola. Tambien puedes indicar otra URL:

```powershell
dotnet run -- web --url http://localhost:5050
```

## Opciones Utiles

```powershell
dotnet run -- train --epochs 80 --batch-size 32
dotnet run -- train --arch resnet101 --learning-rate 0.005
dotnet run -- test --preview-count 20
dotnet run -- web --model-path models/dog-cat-pretrained.zip
```

Arquitecturas disponibles: `resnet50`, `resnet101`, `mobilenet`, `inception`.

## Archivos

```text
ImageClassifier.csproj    dependencias .NET, ML.NET y ASP.NET Core
src/Program.cs            entrada del programa
src/Shared.cs             opciones CLI y carga de datos
src/TrainPretrained.cs    entrenamiento, evaluacion e inferencia
src/WebApp.cs             app web para subir imagenes
```
