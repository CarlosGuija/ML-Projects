# Sentiment Classifier

This project trains and evaluates a sentiment classifier for movie reviews using TensorFlow/Keras.

The current pipeline reads the IMDB dataset, preprocesses the review text, trains a neural network with a `TextVectorization` layer, evaluates it on a held-out test split, and saves the trained model under `models/`.
Each training run also appends the final test metrics to `training_log.csv`.

## Project Structure

```text
sentiment-classifier/
+-- src/
|   +-- data_preprocessing.py  # Load, clean, and split the dataset
|   +-- model.py               # Model architecture
|   +-- train.py               # Training pipeline
+-- data/
|   +-- raw/
|   |   +-- IMDB_Dataset.csv   # Raw IMDB dataset
+-- models/                    # Saved trained models
+-- training_log.csv           # Historical test metrics for training runs
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
7. Appends the model path, test loss, and test accuracy to `training_log.csv`.

The log is a CSV with one row per training run:

```text
timestamp,model_path,test_loss,test_accuracy,epochs_requested,epochs_trained
```

Example:

```csv
timestamp,model_path,test_loss,test_accuracy,epochs_requested,epochs_trained
20260521_163432,models/sentiment_model_20260521_163432.keras,0.2741,0.8920,15,8
```

## Typical Workflow

For normal use, run only:

```powershell
.\.venv\Scripts\python.exe src/train.py
```
