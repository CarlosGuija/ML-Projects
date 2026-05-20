import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text_series(text_series):
    text_series = text_series.astype(str)
    text_series = text_series.str.replace(r'<br\s*/?>', ' ', regex=True, case=False)
    text_series = text_series.str.replace(r'<[^>]+>', ' ', regex=True)
    text_series = text_series.str.replace(r'http\S+|www\S+|https\S+', ' ', regex=True)
    text_series = text_series.str.replace(r'[\x00-\x1f\x7f-\x9f]', ' ', regex=True)
    text_series = text_series.str.replace(r'\s+', ' ', regex=True)
    return text_series.str.strip()

def preprocess_data(data):
    data['cleaned_text'] = clean_text_series(data['text'])
    return data

def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        stratify=data['label']
    )
    return train_data, test_data

def load_and_preprocess_data(file_path):
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    train_data, test_data = split_data(processed_data)
    return train_data, test_data
