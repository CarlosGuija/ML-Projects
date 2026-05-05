import os
import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import SentimentModel
from data_preprocessing import load_and_preprocess_data
from utils import calculate_accuracy, calculate_precision, calculate_recall, save_model

def main():
    train_data, test_data = load_and_preprocess_data('../data/raw/dataset.csv')

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_data['cleaned_text'])

    X_train = tokenizer.texts_to_sequences(train_data['cleaned_text'])
    X_test = tokenizer.texts_to_sequences(test_data['cleaned_text'])

    maxlen = 100
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['label'])
    y_test = encoder.transform(test_data['label'])

    num_classes = len(encoder.classes_)

    model = SentimentModel(input_shape=maxlen, num_classes=num_classes)
    model.build_model()
    model.train(X_train, y_train, batch_size=32, epochs=10)

    y_prob = model.model.predict(X_test)
    y_pred = y_prob.argmax(axis=1)

    print("Accuracy:", calculate_accuracy(y_test, y_pred))
    print("Precision:", calculate_precision(y_test, y_pred))
    print("Recall:", calculate_recall(y_test, y_pred))

    models_dir = '../models'
    if os.path.isfile(models_dir):
        os.remove(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model(model.model, f'{models_dir}/sentiment_model_{timestamp}.h5')

if __name__ == "__main__":
    main()