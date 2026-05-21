# Sentiment Classifier

This project trains and evaluates a sentiment classifier for movie reviews using TensorFlow/Keras.

The current pipeline reads the IMDB dataset, preprocesses the review text, trains a neural network with a `TextVectorization` layer, evaluates it on a held-out test split, and saves the trained model under `models/`.

## Project Structure

```text
sentiment-classifier/
+-- src/
|   +-- data_preprocessing.py  # Load, clean, and split the dataset
|   +-- evaluate_model.py      # Evaluate a saved model
|   +-- model.py               # Model architecture
|   +-- train.py               # Training pipeline
+-- data/
|   +-- raw/
|   |   +-- IMDB_Dataset.csv   # Raw IMDB dataset
|   +-- processed/             # Optional evaluation artifacts
+-- models/                    # Saved trained models
+-- requirements.txt
+-- README.md
```

## Setup

From the project root:

```powershell
pip install -r requirements.txt
```

If you are using the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train a Model

Run:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

This does the full training flow:

1. Loads `data/raw/IMDB_Dataset.csv`.
2. Cleans the text.
3. Creates train, validation, and test splits.
4. Trains the model.
5. Evaluates once on the held-out test set.
6. Saves the trained model to `models/` as a timestamped `.keras` file.

By default, `train.py` only saves the model. It does not write extra processed datasets.

## Save Evaluation Artifacts

Use this only if you want to evaluate the saved model later with `evaluate_model.py`:

```powershell
.\.venv\Scripts\python.exe src/train.py --save-evaluation-artifacts
```

This still saves the model in `models/`, and additionally writes:

```text
data/processed/test_dataset.csv
data/processed/label_classes.json
```

These files let `evaluate_model.py` reuse the same held-out test dataset and label mapping.

## Evaluate a Saved Model

After training with `--save-evaluation-artifacts`, evaluate a saved model like this:

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py models/sentiment_model_YYYYMMDD_HHMMSS.keras
```

Example:

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py models/sentiment_model_20260520_182924.keras
```

By default, `evaluate_model.py` reads:

```text
data/processed/test_dataset.csv
data/processed/label_classes.json
```

You can override those paths:

```powershell
.\.venv\Scripts\python.exe src/evaluate_model.py models/sentiment_model_YYYYMMDD_HHMMSS.keras --test-data-path data/processed/test_dataset.csv --label-classes-path data/processed/label_classes.json
```

## Typical Workflow

For normal use, run only:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

Use `evaluate_model.py` when you want to reload a saved model later and compare its metrics without retraining. In that case, train once with:

```powershell
.\.venv\Scripts\python.exe src/train.py --save-evaluation-artifacts
```
