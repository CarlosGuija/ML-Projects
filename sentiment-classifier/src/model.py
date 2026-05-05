class SentimentModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

        self.model = Sequential()
        self.model.add(Embedding(input_dim=5000, output_dim=128, input_length=self.input_shape))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, batch_size, epochs):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    def save(self, filepath):
        self.model.save(filepath)