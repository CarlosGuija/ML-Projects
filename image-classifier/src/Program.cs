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
    case Command.TrainPretrained:
        TrainPretrained.Run(mlContext, options);
        break;
    case Command.PredictPretrained:
        TrainPretrained.Predict(mlContext, options);
        break;
}
