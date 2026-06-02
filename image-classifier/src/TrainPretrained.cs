using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using static Microsoft.ML.DataOperationsCatalog;

static class TrainPretrained
{
    public static void Run(MLContext mlContext, CliOptions options)
    {
        var trainDir = Path.GetFullPath(Path.Combine(options.DataDir, "train"));
        var testDir = Path.GetFullPath(Path.Combine(options.DataDir, "test"));

        Dataset.EnsureDirectoryExists(trainDir, "directorio de entrenamiento");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.ModelPath))!);
        Directory.CreateDirectory(options.WorkspacePath);

        var trainImages = Dataset.ApplyMaxImagesPerClass(
            Dataset.LoadImagesFromDirectory(trainDir),
            options.MaxImagesPerClass,
            options.Seed);
        if (trainImages.Length == 0)
        {
            throw new InvalidOperationException($"No se encontraron imagenes JPG/PNG en {trainDir}.");
        }

        Console.WriteLine($"Imagenes de entrenamiento encontradas: {trainImages.Length}");
        Dataset.PrintClassBalance(trainImages);

        var trainData = mlContext.Data.LoadFromEnumerable(trainImages);
        var shuffledData = mlContext.Data.ShuffleRows(trainData, seed: options.Seed);
        var preprocessing = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: nameof(ImageData.Label),
                outputColumnName: nameof(PretrainedInput.LabelAsKey))
            .Append(mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: nameof(PretrainedInput.Image),
                imageFolder: "",
                inputColumnName: nameof(ImageData.ImagePath)));

        var preprocessingModel = preprocessing.Fit(shuffledData);
        var preprocessedTrainingData = preprocessingModel.Transform(shuffledData);
        TrainTestData validationSplit = mlContext.Data.TrainTestSplit(
            preprocessedTrainingData,
            testFraction: options.ValidationFraction,
            seed: options.Seed);

        var reportedEpoch = 0;
        var trainerOptions = new ImageClassificationTrainer.Options
        {
            FeatureColumnName = nameof(PretrainedInput.Image),
            LabelColumnName = nameof(PretrainedInput.LabelAsKey),
            ValidationSet = validationSplit.TestSet,
            Arch = options.Architecture,
            Epoch = options.Epochs,
            BatchSize = options.BatchSize,
            LearningRate = options.LearningRate,
            WorkspacePath = options.WorkspacePath,
            MetricsCallback = metrics =>
            {
                reportedEpoch++;
                Console.WriteLine();
                Console.WriteLine($"Epoch {reportedEpoch}/{options.Epochs}");
                Console.WriteLine(metrics);
            },
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        var trainingPipeline = mlContext.MulticlassClassification.Trainers
            .ImageClassification(trainerOptions)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: nameof(PretrainedOutput.PredictedLabelValue),
                inputColumnName: "PredictedLabel"));

        Console.WriteLine("Entrenando modelo preentrenado con transfer learning...");
        Console.WriteLine($"Configuracion: epochs={options.Epochs}, batch-size={options.BatchSize}, arquitectura={options.Architecture}");
        Console.WriteLine("La primera ejecucion puede tardar: ML.NET calcula y guarda cache de caracteristicas antes de imprimir metricas por epoch.");

        using var progressTimer = new Timer(
            _ => Console.WriteLine($"Sigue entrenando... {DateTime.Now:HH:mm:ss}. Si aun no ves epochs, ML.NET sigue preparando/cacheando caracteristicas."),
            null,
            TimeSpan.FromSeconds(30),
            TimeSpan.FromSeconds(60));

        var model = trainingPipeline.Fit(validationSplit.TrainSet);
        progressTimer.Change(Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);

        mlContext.Model.Save(model, preprocessedTrainingData.Schema, options.ModelPath);
        Console.WriteLine($"Modelo preentrenado guardado en: {options.ModelPath}");

        if (Directory.Exists(testDir))
        {
            Evaluate(mlContext, preprocessingModel, model, testDir, options);
        }
    }

    public static void Predict(MLContext mlContext, CliOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.ImagePath))
        {
            throw new ArgumentException("Debes indicar --image para predecir.");
        }

        if (!File.Exists(options.ModelPath))
        {
            throw new FileNotFoundException("No se encontro el modelo entrenado.", options.ModelPath);
        }

        if (!File.Exists(options.ImagePath))
        {
            throw new FileNotFoundException("No se encontro la imagen.", options.ImagePath);
        }

        var model = mlContext.Model.Load(options.ModelPath, out _);
        var predictor = mlContext.Model.CreatePredictionEngine<PretrainedInput, PretrainedOutput>(model);
        var prediction = predictor.Predict(new PretrainedInput
        {
            ImagePath = Path.GetFullPath(options.ImagePath),
            Image = File.ReadAllBytes(options.ImagePath),
            Label = "cats"
        });

        var confidence = prediction.Score.Length == 0 ? 0 : prediction.Score.Max();
        Console.WriteLine($"Prediccion: {Dataset.TranslateLabel(prediction.PredictedLabelValue)}");
        Console.WriteLine($"Confianza: {confidence:0.000}");
    }

    private static void Evaluate(
        MLContext mlContext,
        ITransformer preprocessingModel,
        ITransformer model,
        string testDir,
        CliOptions options)
    {
        var testImages = Dataset.LoadImagesFromDirectory(testDir);
        if (testImages.Length == 0)
        {
            Console.WriteLine($"No se encontraron imagenes JPG/PNG en {testDir}.");
            return;
        }

        Console.WriteLine($"Imagenes de test encontradas: {testImages.Length}");
        var testData = mlContext.Data.LoadFromEnumerable(testImages);
        var preprocessedTestData = preprocessingModel.Transform(testData);
        var predictions = model.Transform(preprocessedTestData);
        var metrics = mlContext.MulticlassClassification.Evaluate(
            predictions,
            labelColumnName: nameof(PretrainedInput.LabelAsKey),
            predictedLabelColumnName: "PredictedLabel");

        Console.WriteLine();
        Console.WriteLine("Evaluacion final en test, modelo preentrenado:");
        Console.WriteLine($"  MicroAccuracy: {metrics.MicroAccuracy:0.0000}");
        Console.WriteLine($"  MacroAccuracy: {metrics.MacroAccuracy:0.0000}");
        Console.WriteLine($"  LogLoss:       {metrics.LogLoss:0.0000}");

        var samplePredictions = mlContext.Data
            .CreateEnumerable<PretrainedOutput>(predictions, reuseRowObject: false)
            .Take(options.PreviewCount);

        Console.WriteLine();
        Console.WriteLine("Muestra de predicciones:");
        foreach (var prediction in samplePredictions)
        {
            Console.WriteLine($"  {Path.GetFileName(prediction.ImagePath)} -> real={Dataset.TranslateLabel(prediction.Label)}, pred={Dataset.TranslateLabel(prediction.PredictedLabelValue)}");
        }
    }
}
