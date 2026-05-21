import argparse
import datetime
import json
from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization

from model import SentimentModel
from data_preprocessing import load_and_preprocess_data

DATA_PATH = Path('data/raw/IMDB_Dataset.csv')
PROCESSED_DATA_DIR = Path('data/processed')
MODELS_DIR = Path('models')
MAX_TOKENS = 50000
BATCH_SIZE = 128
EPOCHS = 15

def parse_args():
    parser = argparse.ArgumentParser(description='Train the sentiment classifier.')
    parser.add_argument(
        '--save-evaluation-artifacts',
        action='store_true',
        help='Save the held-out test dataset and label classes under data/processed.'
    )
    return parser.parse_args()

def build_dataset(texts, labels, batch_size=64, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def save_evaluation_artifacts(test_data, encoder, output_dir=PROCESSED_DATA_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data.to_csv(output_dir / 'test_dataset.csv', index=False)
    with open(output_dir / 'label_classes.json', 'w') as label_file:
        json.dump(encoder.classes_.tolist(), label_file)

def main():
    args = parse_args()

    train_data, test_data = load_and_preprocess_data(DATA_PATH)
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42,
        stratify=train_data['label']
    )

    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        standardize='lower_and_strip_punctuation',
        ngrams=2,
        output_mode='tf_idf'
    )
    vectorizer.adapt(train_data['cleaned_text'].astype(str).to_numpy())

    X_train = train_data['cleaned_text'].astype(str).to_numpy()
    X_val = val_data['cleaned_text'].astype(str).to_numpy()
    X_test = test_data['cleaned_text'].astype(str).to_numpy()

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['label'])
    y_val = encoder.transform(val_data['label'])
    y_test = encoder.transform(test_data['label'])
    if args.save_evaluation_artifacts:
        save_evaluation_artifacts(test_data, encoder)

    num_classes = len(encoder.classes_)

    train_ds = build_dataset(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = build_dataset(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_ds = build_dataset(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]

    model = SentimentModel(
        num_classes=num_classes,
        vectorizer=vectorizer
    )
    model.build_model()
    model.train(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Evaluar solo al final con data no usada durante entrenamiento ni validacion.
    print('\nEvaluating on the test dataset...')
    test_loss, test_accuracy = model.model.evaluate(test_ds, verbose=2)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

    # Guardar el modelo en formato Keras nativo.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model.model.save(MODELS_DIR / f'sentiment_model_{timestamp}.keras')

if __name__ == "__main__":
    main()
