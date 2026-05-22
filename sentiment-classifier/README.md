# Sentiment Classifier

Miniapp and command-line tools for training and using a TensorFlow/Keras sentiment classifier for movie reviews.

The recommended way to use the project is the Streamlit app. The command-line prediction script is still available for quick tests or automation.

## Quick Start

From the project root, install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then open the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit should open the browser automatically. If it does not, go to:

```text
http://localhost:8501
```

## Project Structure

```text
sentiment-classifier/
+-- app.py                         # Streamlit miniapp
+-- requirements.txt               # Python dependencies
+-- training_log.csv               # Training run history
+-- .streamlit/
|   +-- config.toml                 # Streamlit launch settings
+-- src/
|   +-- data_preprocessing.py       # Load and clean data
|   +-- model.py                    # Neural network architecture
|   +-- predict.py                  # Command-line prediction and shared prediction functions
|   +-- train.py                    # Training pipeline
+-- data/
|   +-- raw/                        # Local training dataset, ignored by git
|   |   +-- IMDB_Dataset.csv
|   +-- external/                   # Local CSV files for prediction, ignored by git
|   +-- predictions/                # Local prediction outputs, ignored by git
+-- models/                         # Local saved Keras models, ignored by git
```

Some local files are intentionally ignored by git because they can be large or machine-specific:

- `data/raw/IMDB_Dataset.csv`
- `models/`
- CSV files under `data/external/`
- CSV files under `data/predictions/`
- `.venv/`

## Recommended Use: Streamlit App

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app lets you choose a trained model from `models/` and predict sentiment in two ways.

`Texto`: write one review/comment in the text box and press the prediction button.

`Database`: upload a CSV file, or type a local CSV path. The CSV must have a column named `text`.

For CSV/database predictions, the app shows the results in the browser and lets you download them as a CSV.

## Train a Model

The training script expects the IMDB dataset at:

```text
data/raw/IMDB_Dataset.csv
```

Run:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

Training does this:

1. Loads and cleans the IMDB dataset.
2. Creates train, validation, and test splits.
3. Builds a model with a `TextVectorization` layer and dense neural network.
4. Trains with early stopping and learning-rate reduction.
5. Evaluates once on the test set.
6. Saves the model in `models/` as `sentiment_model_YYYYMMDD_HHMMSS.keras`.
7. Saves label metadata next to the model as `sentiment_model_YYYYMMDD_HHMMSS.json`.
8. Appends the final metrics to `training_log.csv`.

`training_log.csv` has one row per training run:

```text
timestamp,model_path,test_loss,test_accuracy,epochs_requested,epochs_trained
```

## Alternative: Command-Line Prediction

Use `src/predict.py` when you want to predict from the terminal instead of opening the app.

### Predict One Text

```powershell
.\.venv\Scripts\python.exe src/predict.py --text "This movie was surprisingly good"
```

By default, this uses the latest saved model in `models/`.

To use a specific model:

```powershell
.\.venv\Scripts\python.exe src/predict.py --model models/sentiment_model_YYYYMMDD_HHMMSS.keras --text "This movie was surprisingly good"
```

### Predict a CSV

The CSV must contain a `text` column:

```csv
text
"This movie was surprisingly good"
"I hated the ending"
```

Print predictions in the terminal without saving a file:

```powershell
.\.venv\Scripts\python.exe src/predict.py --input data/external/new_reviews.csv
```

Save predictions only when you explicitly pass `--output`:

```powershell
.\.venv\Scripts\python.exe src/predict.py --input data/external/new_reviews.csv --output data/predictions/new_reviews_predictions.csv
```

The output includes the original columns plus:

- `prediction`
- one probability column per class, for example `negative_probability` and `positive_probability`

## Typical Workflow

1. Put `IMDB_Dataset.csv` in `data/raw/`.
2. Train a model:

```powershell
.\.venv\Scripts\python.exe src/train.py
```

3. Open the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Predict using `Texto` or `Database`.
