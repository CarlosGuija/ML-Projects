using Microsoft.ML;

var options = CliOptions.Parse(args);
if (options.ShowHelp)
{
    CliOptions.PrintHelp();
    return;
}

var mlContext = new MLContext(seed: options.Seed);

switch (options.Command)
{
    case Command.Train:
        TrainPretrained.Run(mlContext, options);
        break;
    case Command.Evaluate:
        TrainPretrained.EvaluateSavedModel(mlContext, options);
        break;
    case Command.Predict:
        TrainPretrained.Predict(mlContext, options);
        break;
    case Command.Web:
        WebApp.Run(mlContext, options);
        break;
}
