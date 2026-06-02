using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System.Globalization;

enum Command
{
    TrainPretrained,
    PredictPretrained
}

static class Defaults
{
    public const string DataDir = "data/raw";
    public const string PretrainedModelPath = "models/dog-cat-pretrained.zip";
    public const string WorkspacePath = "outputs/mlnet-cache";
}

class ImageData
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
}

sealed class PretrainedInput : ImageData
{
    public byte[] Image { get; set; } = [];
    public uint LabelAsKey { get; set; }
}

sealed class PretrainedOutput : ImageData
{
    public string PredictedLabelValue { get; set; } = "";
    public float[] Score { get; set; } = [];
}

sealed class CliOptions
{
    public Command Command { get; private init; } = Command.TrainPretrained;
    public string DataDir { get; private init; } = Defaults.DataDir;
    public string ModelPath { get; private init; } = Defaults.PretrainedModelPath;
    public string WorkspacePath { get; private init; } = Defaults.WorkspacePath;
    public string? ImagePath { get; private init; }
    public int Epochs { get; private init; } = 80;
    public int BatchSize { get; private init; } = 32;
    public int Seed { get; private init; } = 42;
    public int PreviewCount { get; private init; } = 10;
    public int? MaxImagesPerClass { get; private init; }
    public float LearningRate { get; private init; } = 0.01f;
    public double ValidationFraction { get; private init; } = 0.15;
    public bool ShowHelp { get; private init; }
    public ImageClassificationTrainer.Architecture Architecture { get; private init; } =
        ImageClassificationTrainer.Architecture.ResnetV250;

    public static CliOptions Parse(string[] args)
    {
        var parsed = new MutableCliOptions();
        var index = 0;

        if (args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal))
        {
            parsed.Command = ParseCommand(args[0]);
            index = 1;
        }

        while (index < args.Length)
        {
            var name = args[index++];
            if (name is "--help" or "-h")
            {
                parsed.ShowHelp = true;
                continue;
            }

            if (index >= args.Length)
            {
                throw new ArgumentException($"Falta valor para {name}.");
            }

            var value = args[index++];
            switch (name)
            {
                case "--data-dir":
                    parsed.DataDir = value;
                    break;
                case "--model-path":
                    parsed.ModelPath = value;
                    break;
                case "--workspace-path":
                    parsed.WorkspacePath = value;
                    break;
                case "--image":
                    parsed.ImagePath = value;
                    break;
                case "--epochs":
                    parsed.Epochs = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--batch-size":
                    parsed.BatchSize = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--max-images-per-class":
                    parsed.MaxImagesPerClass = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--learning-rate":
                    parsed.LearningRate = float.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--validation-fraction":
                    parsed.ValidationFraction = double.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--seed":
                    parsed.Seed = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--preview-count":
                    parsed.PreviewCount = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "--arch":
                    parsed.Architecture = ParseArchitecture(value);
                    break;
                default:
                    throw new ArgumentException($"Opcion no reconocida: {name}");
            }
        }

        return parsed.ToImmutable();
    }

    public static void PrintHelp()
    {
        Console.WriteLine("""
        Clasificador perros vs gatos en C# con ML.NET.

        Uso:
          dotnet run -- train [opciones]
          dotnet run -- predict --image ruta/a/imagen.jpg

        Opciones:
          --data-dir <ruta>          Directorio raiz con train/ y test/. Default: data/raw
          --model-path <ruta>        Ruta del modelo .zip.
          --epochs <n>               Epocas. Default pretrained: 80
          --batch-size <n>           Tamano del batch. Default: 32
          --max-images-per-class <n> Usa solo n imagenes por clase para probar rapido.
          --learning-rate <valor>    Learning rate. Default: 0.01
          --validation-fraction <v>  Fraccion de train usada para validacion. Default: 0.15
          --arch <resnet50|resnet101|mobilenet|inception>
          --help                     Muestra esta ayuda.
        """);
    }

    private static Command ParseCommand(string value) => value.ToLowerInvariant() switch
    {
        "train" or "train-pretrained" => Command.TrainPretrained,
        "predict" or "predict-pretrained" => Command.PredictPretrained,
        _ => throw new ArgumentException($"Comando no reconocido: {value}")
    };

    private static ImageClassificationTrainer.Architecture ParseArchitecture(string value) =>
        value.ToLowerInvariant() switch
        {
            "resnet50" or "resnetv250" => ImageClassificationTrainer.Architecture.ResnetV250,
            "resnet101" or "resnetv2101" => ImageClassificationTrainer.Architecture.ResnetV2101,
            "mobilenet" or "mobilenetv2" => ImageClassificationTrainer.Architecture.MobilenetV2,
            "inception" or "inceptionv3" => ImageClassificationTrainer.Architecture.InceptionV3,
            _ => throw new ArgumentException($"Arquitectura no reconocida: {value}")
        };

    private sealed class MutableCliOptions
    {
        public Command Command { get; set; } = Command.TrainPretrained;
        public string DataDir { get; set; } = Defaults.DataDir;
        public string ModelPath { get; set; } = Defaults.PretrainedModelPath;
        public string WorkspacePath { get; set; } = Defaults.WorkspacePath;
        public string? ImagePath { get; set; }
        public int Epochs { get; set; } = 80;
        public int BatchSize { get; set; } = 32;
        public int Seed { get; set; } = 42;
        public int PreviewCount { get; set; } = 10;
        public int? MaxImagesPerClass { get; set; }
        public float LearningRate { get; set; } = 0.01f;
        public double ValidationFraction { get; set; } = 0.15;
        public bool ShowHelp { get; set; }
        public ImageClassificationTrainer.Architecture Architecture { get; set; } =
            ImageClassificationTrainer.Architecture.ResnetV250;

        public CliOptions ToImmutable() => new()
        {
            Command = Command,
            DataDir = DataDir,
            ModelPath = ModelPath,
            WorkspacePath = WorkspacePath,
            ImagePath = ImagePath,
            Epochs = Epochs,
            BatchSize = BatchSize,
            Seed = Seed,
            PreviewCount = PreviewCount,
            MaxImagesPerClass = MaxImagesPerClass,
            LearningRate = LearningRate,
            ValidationFraction = ValidationFraction,
            ShowHelp = ShowHelp,
            Architecture = Architecture
        };
    }
}

static class Dataset
{
    public static ImageData[] LoadImagesFromDirectory(string folder)
    {
        var supportedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            ".jpg",
            ".jpeg",
            ".png"
        };

        var skippedUnsupportedImages = 0;
        var images = new List<ImageData>();

        foreach (var file in Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories))
        {
            if (!supportedExtensions.Contains(Path.GetExtension(file)))
            {
                continue;
            }

            if (!HasJpegOrPngSignature(file))
            {
                skippedUnsupportedImages++;
                continue;
            }

            var label = Directory.GetParent(file)?.Name ?? "";
            if (string.IsNullOrWhiteSpace(label))
            {
                continue;
            }

            images.Add(new ImageData
            {
                ImagePath = Path.GetFullPath(file),
                Label = label,
                });
        }

        if (skippedUnsupportedImages > 0)
        {
            Console.WriteLine($"Aviso: se ignoraron {skippedUnsupportedImages} archivos con extension JPG/PNG pero contenido no compatible. Probablemente BMP o imagenes corruptas.");
        }

        return images.ToArray();
    }

    private static bool HasJpegOrPngSignature(string file)
    {
        Span<byte> header = stackalloc byte[8];

        try
        {
            using var stream = File.OpenRead(file);
            var bytesRead = stream.Read(header);

            var isJpeg = bytesRead >= 3 &&
                header[0] == 0xFF &&
                header[1] == 0xD8 &&
                header[2] == 0xFF;

            var isPng = bytesRead >= 8 &&
                header[0] == 0x89 &&
                header[1] == 0x50 &&
                header[2] == 0x4E &&
                header[3] == 0x47 &&
                header[4] == 0x0D &&
                header[5] == 0x0A &&
                header[6] == 0x1A &&
                header[7] == 0x0A;

            return isJpeg || isPng;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    public static void EnsureDirectoryExists(string path, string description)
    {
        if (!Directory.Exists(path))
        {
            throw new DirectoryNotFoundException($"No se encontro el {description}: {path}");
        }
    }

    public static void PrintClassBalance(IEnumerable<ImageData> images)
    {
        foreach (var group in images.GroupBy(image => image.Label).OrderBy(group => group.Key))
        {
            Console.WriteLine($"  {TranslateLabel(group.Key)}: {group.Count()}");
        }
    }

    public static ImageData[] ApplyMaxImagesPerClass(
        IEnumerable<ImageData> images,
        int? maxImagesPerClass,
        int seed)
    {
        var allImages = images.ToArray();
        if (maxImagesPerClass is null)
        {
            return allImages;
        }

        var random = new Random(seed);
        return allImages
            .GroupBy(image => image.Label)
            .SelectMany(group => group
                .OrderBy(_ => random.Next())
                .Take(maxImagesPerClass.Value))
            .ToArray();
    }

    public static string TranslateLabel(string? label) => label?.ToLowerInvariant() switch
    {
        "cats" or "cat" => "gato",
        "dogs" or "dog" => "perro",
        null or "" => "desconocido",
        _ => label
    };
}
