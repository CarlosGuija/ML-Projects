using Microsoft.ML;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;

static class WebApp
{
    public static void Run(MLContext mlContext, CliOptions options)
    {
        var appUrl = ResolveAvailableUrl(options.Url);
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls(appUrl);

        var app = builder.Build();
        app.MapGet("/", () => Results.Content(RenderPage(), "text/html"));
        app.MapPost("/predict", async (HttpRequest request) =>
        {
            var modelPath = Path.GetFullPath(options.ModelPath);
            if (!File.Exists(modelPath))
            {
                return Results.Content(
                    RenderPage(error: $"No se encontro el modelo en {modelPath}. Entrena primero con dotnet run -- train."),
                    "text/html");
            }

            var form = await request.ReadFormAsync();
            var image = form.Files["image"];
            if (image is null || image.Length == 0)
            {
                return Results.Content(RenderPage(error: "Sube una imagen JPG o PNG."), "text/html");
            }

            var extension = Path.GetExtension(image.FileName).ToLowerInvariant();
            if (extension is not ".jpg" and not ".jpeg" and not ".png")
            {
                return Results.Content(RenderPage(error: "Formato no soportado. Usa JPG o PNG."), "text/html");
            }

            var uploadDir = Path.Combine(Path.GetTempPath(), "image-classifier-uploads");
            Directory.CreateDirectory(uploadDir);
            var imagePath = Path.Combine(uploadDir, $"{Guid.NewGuid():N}{extension}");

            byte[] imageBytes;
            await using (var stream = File.Create(imagePath))
            {
                await image.CopyToAsync(stream);
            }

            try
            {
                imageBytes = await File.ReadAllBytesAsync(imagePath);
                var model = mlContext.Model.Load(modelPath, out _);
                var prediction = TrainPretrained.PredictImage(mlContext, model, imagePath);
                var imageDataUrl = $"data:{image.ContentType};base64,{Convert.ToBase64String(imageBytes)}";

                return Results.Content(RenderPage(prediction, image.FileName, imageDataUrl), "text/html");
            }
            finally
            {
                File.Delete(imagePath);
            }
        });

        if (!string.Equals(appUrl, options.Url, StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"El puerto de {options.Url} esta ocupado. Usando {appUrl}.");
        }

        Console.WriteLine($"App web disponible en {appUrl}");
        app.Lifetime.ApplicationStarted.Register(() => OpenBrowser(appUrl));
        app.Run();
    }

    private static string ResolveAvailableUrl(string preferredUrl)
    {
        var uri = new Uri(preferredUrl);
        var builder = new UriBuilder(uri);
        var startingPort = uri.Port;

        for (var port = startingPort; port < startingPort + 20; port++)
        {
            if (IsPortAvailable(port))
            {
                builder.Port = port;
                return builder.Uri.ToString().TrimEnd('/');
            }
        }

        return preferredUrl;
    }

    private static bool IsPortAvailable(int port)
    {
        return CanListen(IPAddress.Loopback, port) && CanListen(IPAddress.IPv6Loopback, port);
    }

    private static bool CanListen(IPAddress address, int port)
    {
        try
        {
            using var listener = new TcpListener(address, port);
            listener.Start();
            return true;
        }
        catch (SocketException)
        {
            return false;
        }
    }

    private static void OpenBrowser(string url)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = url,
                UseShellExecute = true
            });
        }
        catch (Exception exception)
        {
            Console.WriteLine($"No pude abrir el navegador automaticamente: {exception.Message}");
            Console.WriteLine($"Abre manualmente: {url}");
        }
    }

    private static string RenderPage(
        PredictionResponse? prediction = null,
        string? fileName = null,
        string? imageDataUrl = null,
        string? error = null)
    {
        var resultHtml = prediction is null
            ? ""
            : $"""
              <section class="result">
                <div>
                  <p class="eyebrow">Prediccion</p>
                  <h2>{Encode(prediction.PredictedLabel)}</h2>
                  <p>Confianza: {Percent(prediction.Confidence)}</p>
                </div>
                <div class="bars">
                  {RenderProbability("gato", prediction)}
                  {RenderProbability("perro", prediction)}
                </div>
              </section>
              """;

        var previewHtml = imageDataUrl is null
            ? ""
            : $"""
              <section class="preview">
                <img src="{imageDataUrl}" alt="{Encode(fileName ?? "Imagen subida")}">
              </section>
              """;

        var errorHtml = string.IsNullOrWhiteSpace(error)
            ? ""
            : $"""<p class="error">{Encode(error)}</p>""";

        return $$"""
        <!doctype html>
        <html lang="es">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Clasificador gato o perro</title>
          <style>
            :root {
              color-scheme: light;
              font-family: Arial, Helvetica, sans-serif;
              background: #f7f8fa;
              color: #1f2933;
            }

            body {
              margin: 0;
              min-height: 100vh;
            }

            main {
              width: min(900px, calc(100% - 32px));
              margin: 0 auto;
              padding: 40px 0;
            }

            h1 {
              margin: 0 0 8px;
              font-size: 34px;
              font-weight: 700;
            }

            p {
              margin: 0;
              line-height: 1.5;
            }

            form,
            .result,
            .preview {
              margin-top: 24px;
              border: 1px solid #d8dee7;
              border-radius: 8px;
              background: #ffffff;
              padding: 20px;
            }

            input[type="file"] {
              display: block;
              width: 100%;
              margin: 16px 0;
            }

            button {
              border: 0;
              border-radius: 6px;
              background: #12665f;
              color: #ffffff;
              padding: 10px 16px;
              font-size: 15px;
              cursor: pointer;
            }

            .preview img {
              display: block;
              max-width: 100%;
              max-height: 460px;
              object-fit: contain;
              margin: 0 auto;
            }

            .result {
              display: grid;
              grid-template-columns: minmax(0, 1fr) minmax(260px, 1.4fr);
              gap: 24px;
              align-items: start;
            }

            .result h2 {
              margin: 4px 0 8px;
              font-size: 42px;
            }

            .eyebrow {
              color: #596575;
              font-size: 13px;
              text-transform: uppercase;
            }

            .bar {
              margin-bottom: 16px;
            }

            .bar-label {
              display: flex;
              justify-content: space-between;
              margin-bottom: 6px;
              font-weight: 700;
            }

            .track {
              height: 14px;
              border-radius: 999px;
              background: #e8edf3;
              overflow: hidden;
            }

            .fill {
              height: 100%;
              border-radius: inherit;
              background: #d1495b;
            }

            .error {
              margin-top: 16px;
              color: #a11d2b;
              font-weight: 700;
            }

            @media (max-width: 720px) {
              main {
                padding: 24px 0;
              }

              .result {
                grid-template-columns: 1fr;
              }
            }
          </style>
        </head>
        <body>
          <main>
            <h1>Clasificador gato o perro</h1>
            <p>Sube una foto JPG o PNG y el modelo dira si parece un gato o un perro.</p>
            <form method="post" action="/predict" enctype="multipart/form-data">
              <label for="image">Foto</label>
              <input id="image" name="image" type="file" accept="image/jpeg,image/png" required>
              <button type="submit">Analizar</button>
              {{errorHtml}}
            </form>
            {{previewHtml}}
            {{resultHtml}}
          </main>
        </body>
        </html>
        """;
    }

    private static string RenderProbability(string label, PredictionResponse prediction)
    {
        var value = prediction.Probabilities.TryGetValue(label, out var probability) ? probability : 0;
        var width = Math.Clamp(value * 100, 0, 100).ToString("0.0", CultureInfo.InvariantCulture);

        return $"""
          <div class="bar">
            <div class="bar-label">
              <span>{Encode(label)}</span>
              <span>{Percent(value)}</span>
            </div>
            <div class="track">
              <div class="fill" style="width: {width}%"></div>
            </div>
          </div>
          """;
    }

    private static string Percent(float value) =>
        $"{Math.Clamp(value, 0, 1) * 100:0.0}%";

    private static string Encode(string value) => WebUtility.HtmlEncode(value);
}
