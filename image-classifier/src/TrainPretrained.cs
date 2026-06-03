using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

static class TrainPretrained
{
    public static void Run(MLContext mlContext, CliOptions options)
    {
        var trainDir = Path.GetFullPath(Path.Combine(options.DataDir, "train"));
        var testDir = Path.GetFullPath(Path.Combine(options.DataDir, "test"));

        Dataset.EnsureDirectoryExists(trainDir, "directorio de entrenamiento");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.ModelPath))!);
        Directory.CreateDirectory(options.WorkspacePath);

        var trainImages = Dataset.LoadImagesFromDirectory(
            trainDir,
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
            outputColumnName: nameof(PretrainedInput.LabelAsKey));

        var preprocessingModel = preprocessing.Fit(shuffledData);
        var preprocessedTrainingData = preprocessingModel.Transform(shuffledData);
        DataOperationsCatalog.TrainTestData validationSplit = mlContext.Data.TrainTestSplit(
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

        var model = trainingPipeline.Fit(validationSplit.TrainSet);

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
        var response = PredictImage(mlContext, model, options.ImagePath);

        Console.WriteLine($"Prediccion: {response.PredictedLabel}");
        Console.WriteLine($"Confianza: {response.Confidence:0.000}");
        foreach (var probability in response.Probabilities)
        {
            Console.WriteLine($"Probabilidad {probability.Key}: {probability.Value:0.000}");
        }
    }

    public static PredictionResponse PredictImage(
        MLContext mlContext,
        ITransformer model,
        string imagePath)
    {
        var input = new PretrainedInput
        {
            ImagePath = Path.GetFullPath(imagePath),
            Image = File.ReadAllBytes(imagePath),
            Label = "cats"
        };

        var inputData = mlContext.Data.LoadFromEnumerable([input]);
        var predictions = model.Transform(inputData);
        var prediction = mlContext.Data
            .CreateEnumerable<PretrainedOutput>(predictions, reuseRowObject: false)
            .Single();
        var probabilities = GetProbabilitiesByLabel(predictions.Schema, prediction.Score);
        var confidence = prediction.Score.Length == 0 ? 0 : prediction.Score.Max();

        return new PredictionResponse(
            Dataset.TranslateLabel(prediction.PredictedLabelValue),
            confidence,
            probabilities);
    }

    public static void EvaluateSavedModel(MLContext mlContext, CliOptions options)
    {
        var testDir = Path.GetFullPath(Path.Combine(options.DataDir, "test"));
        Dataset.EnsureDirectoryExists(testDir, "directorio de test");

        if (!File.Exists(options.ModelPath))
        {
            throw new FileNotFoundException("No se encontro el modelo entrenado.", options.ModelPath);
        }

        Console.WriteLine("Cargando imagenes de test...");
        var testImages = Dataset.LoadImagesFromDirectory(testDir);
        if (testImages.Length == 0)
        {
            Console.WriteLine($"No se encontraron imagenes JPG/PNG en {testDir}.");
            return;
        }

        Console.WriteLine($"Modelo: {options.ModelPath}");
        Console.WriteLine($"Imagenes de test encontradas: {testImages.Length}");
        Dataset.PrintClassBalance(testImages);

        Console.WriteLine("Cargando modelo TensorFlow/ML.NET...");
        var model = mlContext.Model.Load(options.ModelPath, out _);
        Console.WriteLine("Preparando pipeline de inferencia...");
        var testData = mlContext.Data.LoadFromEnumerable(testImages);
        var predictions = model.Transform(testData);
        Console.WriteLine("Generando predicciones y calculando metricas...");
        var predictionRows = mlContext.Data
            .CreateEnumerable<PretrainedOutput>(predictions, reuseRowObject: false)
            .ToArray();

        PrintPredictionReport("Resultado en test:", predictionRows, options.PreviewCount);
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

        var predictionRows = mlContext.Data
            .CreateEnumerable<PretrainedOutput>(predictions, reuseRowObject: false)
            .ToArray();

        Console.WriteLine();
        PrintPredictionReport("Muestra de predicciones:", predictionRows, options.PreviewCount);
    }

    private static void PrintPredictionReport(
        string title,
        IReadOnlyCollection<PretrainedOutput> predictions,
        int previewCount)
    {
        var summary = PredictionSummary.From(predictions);
        var accuracy = predictions.Count == 0 ? 0 : summary.Correct / (double)predictions.Count;

        Console.WriteLine(title);
        foreach (var prediction in predictions.Take(previewCount))
        {
            var confidence = prediction.Score.Length == 0 ? 0 : prediction.Score.Max();
            Console.WriteLine($"  {Path.GetFileName(prediction.ImagePath)} -> real={Dataset.TranslateLabel(prediction.Label)}, pred={Dataset.TranslateLabel(prediction.PredictedLabelValue)}, conf={confidence:0.000}");
        }

        Console.WriteLine();
        Console.WriteLine($"  Accuracy: {accuracy:0.0000}");
        Console.WriteLine($"  Correctas: {summary.Correct}/{predictions.Count}");
        Console.WriteLine();
        Console.WriteLine("Matriz de predicciones:");
        Console.WriteLine("                 pred=gato  pred=perro");
        Console.WriteLine($"  real=gato      {summary.CatAsCat,9}  {summary.CatAsDog,10}");
        Console.WriteLine($"  real=perro     {summary.DogAsCat,9}  {summary.DogAsDog,10}");
        Console.WriteLine("  Correctas en la diagonal.");
    }

    private static string NormalizeLabel(string? label) => label?.ToLowerInvariant() switch
    {
        "cat" or "cats" or "gato" => "cats",
        "dog" or "dogs" or "perro" => "dogs",
        _ => label?.ToLowerInvariant() ?? ""
    };

    private static IReadOnlyDictionary<string, float> GetProbabilitiesByLabel(
        DataViewSchema schema,
        float[] scores)
    {
        if (scores.Length == 0)
        {
            return new Dictionary<string, float>();
        }

        var labels = GetScoreLabels(schema, scores.Length);
        return labels
            .Select((label, index) => new
            {
                Label = Dataset.TranslateLabel(label),
                Score = scores[index]
            })
            .GroupBy(item => item.Label)
            .ToDictionary(group => group.Key, group => group.First().Score);
    }

    private static string[] GetScoreLabels(DataViewSchema schema, int scoreCount)
    {
        var scoreColumn = schema.FirstOrDefault(column => column.Name == nameof(PretrainedOutput.Score));
        if (scoreColumn.Name == nameof(PretrainedOutput.Score))
        {
            var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
            scoreColumn.GetSlotNames(ref slotNames);
            var labels = slotNames
                .DenseValues()
                .Select(label => label.ToString())
                .Where(label => !string.IsNullOrWhiteSpace(label))
                .ToArray();

            if (labels.Length == scoreCount)
            {
                return labels;
            }
        }

        return scoreCount == 2
            ? ["cats", "dogs"]
            : Enumerable.Range(1, scoreCount).Select(index => $"clase_{index}").ToArray();
    }

    private readonly record struct PredictionSummary(
        int Correct,
        int CatAsCat,
        int CatAsDog,
        int DogAsCat,
        int DogAsDog)
    {
        public static PredictionSummary From(IEnumerable<PretrainedOutput> predictions)
        {
            var correct = 0;
            var catAsCat = 0;
            var catAsDog = 0;
            var dogAsCat = 0;
            var dogAsDog = 0;

            foreach (var prediction in predictions)
            {
                var actual = NormalizeLabel(prediction.Label);
                var predicted = NormalizeLabel(prediction.PredictedLabelValue);
                if (actual == predicted)
                {
                    correct++;
                }

                if (actual == "cats" && predicted == "cats")
                {
                    catAsCat++;
                }
                else if (actual == "cats" && predicted == "dogs")
                {
                    catAsDog++;
                }
                else if (actual == "dogs" && predicted == "cats")
                {
                    dogAsCat++;
                }
                else if (actual == "dogs" && predicted == "dogs")
                {
                    dogAsDog++;
                }
            }

            return new PredictionSummary(correct, catAsCat, catAsDog, dogAsCat, dogAsDog);
        }
    }
}

public sealed record PredictionResponse(
    string PredictedLabel,
    float Confidence,
    IReadOnlyDictionary<string, float> Probabilities);
