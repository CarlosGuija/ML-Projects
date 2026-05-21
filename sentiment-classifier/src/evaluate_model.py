import argparse
import json

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from train import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained sentiment model on the held-out test dataset.'
    )
    parser.add_argument(
        'model_path',
        help='Path to the trained .keras model file.'
    )
    parser.add_argument(
        '--test-data-path',
        default='data/processed/test_dataset.csv',
        help='Path to the saved held-out test dataset CSV.'
    )
    parser.add_argument(
        '--label-classes-path',
        default='data/processed/label_classes.json',
        help='Path to the saved label classes JSON file.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size used for evaluation.'
    )
    return parser.parse_args()


def make_test_dataset(test_data_path, label_classes_path, batch_size):
    test_data = pd.read_csv(test_data_path)
    with open(label_classes_path) as label_file:
        label_classes = json.load(label_file)

    encoder = LabelEncoder()
    encoder.classes_ = pd.Series(label_classes).to_numpy()

    x_test = test_data['cleaned_text'].astype(str).to_numpy()
    y_test = encoder.transform(test_data['label'])

    return build_dataset(x_test, y_test, batch_size=batch_size, shuffle=False)


def main():
    args = parse_args()

    model = load_model(args.model_path)
    test_ds = make_test_dataset(
        args.test_data_path,
        args.label_classes_path,
        args.batch_size
    )

    print('\nEvaluating loaded model on the held-out test dataset...')
    results = model.evaluate(test_ds, verbose=2, return_dict=True)

    for metric_name, metric_value in results.items():
        print(f'{metric_name}: {metric_value:.4f}')


if __name__ == '__main__':
    main()
