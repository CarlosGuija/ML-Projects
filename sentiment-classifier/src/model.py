import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input

class SentimentModel:
    def __init__(self, num_classes, vectorizer):
        self.num_classes = num_classes
        self.vectorizer = vectorizer
        self.model = None

    def build_model(self):
        inputs = Input(shape=(), dtype=tf.string)
        x = self.vectorizer(inputs)
        x = Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = Dropout(0.4)(x)
        x = Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, train_ds, validation_data=None, epochs=10, callbacks=None):
        return self.model.fit(
            train_ds,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=2
        )
