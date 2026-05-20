def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct_predictions / len(y_true)

def calculate_precision(y_true, y_pred):
    true_positives = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    predicted_positives = sum(y_p == 1 for y_p in y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def calculate_recall(y_true, y_pred):
    true_positives = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    actual_positives = sum(y_t == 1 for y_t in y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)
