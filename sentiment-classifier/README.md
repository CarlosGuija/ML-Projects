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
|   +-- predict.py             # Predict sentiment for new text
|   +-- train.py               # Training pipeline
+-- data/
|   +-- external/              # New external datasets for prediction
|   +-- predictions/           # Prediction outputs
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

## Recommended Use: Streamlit App

After training at least one model, run the app from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in your browser and lets you choose a saved model from `models/`.
If the browser does not open automatically, go to `http://localhost:8501`.

You can predict sentiment in two ways:

1. `Texto`: write one review/comment in the text box and press the prediction button.
2. `Database`: upload a CSV file, or type a local CSV path, with a `text` column.

For CSV/database predictions, the app shows the results in the browser and lets you download them as a CSV.

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
7. Saves the model label metadata next to the model as a timestamped `.json` file.
8. Appends the model path, test loss, and test accuracy to `training_log.csv`.

The log is a CSV with one row per training run:

```text
timestamp,model_path,test_loss,test_accuracy,epochs_requested,epochs_trained
```

Example:

```csv
timestamp,model_path,test_loss,test_accuracy,epochs_requested,epochs_trained
20260521_163432,models/sentiment_model_20260521_163432.keras,0.2741,0.8920,15,8
```

## Alternative: Command-Line Prediction

You can still use `src/predict.py` directly from the terminal. This is useful for quick checks, scripts, or when you do not want to open the Streamlit app.

### Predict New Text

After training at least one model, run:

```powershell
.\.venv\Scripts\python.exe src/predict.py --text "This movie was surprisingly good"
```

By default, this uses the latest saved model in `models/`.

To use a specific model:

```powershell
.\.venv\Scripts\python.exe src/predict.py --model models/sentiment_model_YYYYMMDD_HHMMSS.keras --text "This movie was surprisingly good"
```

### Predict a Dataset

Save external CSV files under:

```text
data/external/
```

The CSV must contain a `text` column:

```csv
text
"This movie was surprisingly good"
"I hated the ending"
```

Then run:

```powershell
.\.venv\Scripts\python.exe src/predict.py --input data/external/new_reviews.csv
```

By default, this prints the predictions in the terminal and does not save a file.

To save predictions, pass an explicit `--output` path:

```powershell
.\.venv\Scripts\python.exe src/predict.py --input data/external/new_reviews.csv --output data/predictions/new_reviews_predictions.csv
```

The output includes the original columns plus `prediction` and one probability column per class.

## Typical Workflow

For normal use:

```powershell
.\.venv\Scripts\python.exe src/train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
