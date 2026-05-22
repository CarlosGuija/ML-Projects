import argparse
import json
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd

from data_preprocessing import clean_text_series

MODELS_DIR = Path('models')
PREDICTIONS_DIR = Path('data/predictions')
DEFAULT_LABEL_CLASSES = ['negative', 'positive']
TEXT_COLUMN = 'text'


def parse_args():
    parser = argparse.ArgumentParser(description='Predict sentiment for new text.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        help='Raw review text to classify.'
    )
    input_group.add_argument(
        '--input',
        type=Path,
        help='Path to a CSV file with a text column.'
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=None,
        help='Path to a saved .keras model. Defaults to the latest model in models/.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Path for CSV predictions. Defaults to data/predictions/<input-name>_predictions.csv.'
    )
    return parser.parse_args()


def find_latest_model(models_dir=MODELS_DIR):
    model_paths = sorted(models_dir.glob('sentiment_model_*.keras'))
    if not model_paths:
        raise FileNotFoundError(f'No saved models found in {models_dir}.')
    return model_paths[-1]


def load_label_classes(model_path):
    metadata_path = model_path.with_suffix('.json')
    if not metadata_path.exists():
        return DEFAULT_LABEL_CLASSES

    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata['label_classes']


def predict_sentiment(model, label_classes, raw_text):
    cleaned_text = clean_text_series(pd.Series([raw_text])).to_numpy()
    probabilities = model.predict(cleaned_text, verbose=0)[0]
    predicted_index = int(probabilities.argmax())

    return label_classes[predicted_index], probabilities


def predict_dataframe(model, label_classes, input_path):
    data = pd.read_csv(input_path)
    if TEXT_COLUMN not in data.columns:
        raise ValueError(f'Input CSV must contain a "{TEXT_COLUMN}" column.')

    cleaned_text = clean_text_series(data[TEXT_COLUMN]).to_numpy()
    probabilities = model.predict(cleaned_text, verbose=0)
    predicted_indexes = probabilities.argmax(axis=1)

    output_data = data.copy()
    output_data['prediction'] = [label_classes[index] for index in predicted_indexes]
    for index, label in enumerate(label_classes):
        output_data[f'{label}_probability'] = probabilities[:, index]

    return output_data


def default_output_path(input_path):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_DIR / f'{input_path.stem}_predictions.csv'


def main():
    args = parse_args()

    from tensorflow.keras.models import load_model

    model_path = args.model or find_latest_model()
    model = load_model(model_path)
    label_classes = load_label_classes(model_path)

    print(f'Model: {model_path}')

    if args.text:
        prediction, probabilities = predict_sentiment(model, label_classes, args.text)
        print(f'Prediction: {prediction}')
        print('Probabilities:')
        for label, probability in zip(label_classes, probabilities):
            print(f'  {label}: {probability:.4f}')
        return

    output_path = args.output or default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = predict_dataframe(model, label_classes, args.input)
    predictions.to_csv(output_path, index=False)
    print(f'Predictions saved: {output_path}')


if __name__ == '__main__':
    main()
