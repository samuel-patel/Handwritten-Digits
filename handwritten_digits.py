import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, callbacks


class Model:

    def __init__(self, input_shape, num_outputs):
        # Dropout is used to prevent overfitting and batch normalization is used to aid in training
        self.neural_net = keras.Sequential([
            layers.Input(input_shape),
            layers.BatchNormalization(),
            layers.Dense(units=16, activation="relu"),
            layers.Dropout(rate=0.3),
            layers.BatchNormalization(),
            layers.Dense(units=16, activation="relu"),
            layers.Dropout(rate=0.3),
            layers.BatchNormalization(),
            layers.Dense(units=num_outputs, activation="sigmoid")
        ])

        self.history = None

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.neural_net.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stopping = callbacks.EarlyStopping(
            min_delta=0.001,
            patience=20,
            restore_best_weights=True,
        )

        X_train, y_train = self.preprocess_data(X_train, y_train)
        X_valid, y_valid = self.preprocess_data(X_valid, y_valid)

        history_object = self.neural_net.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=256,
            epochs=100,
            callbacks=[early_stopping]
        )

        self.history = pd.DataFrame(history_object.history)

    def plot_iterations(self):
        self.history.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    def predict(self, X):
        return [np.argmax(i) for i in self.neural_net.predict(X)]

    @staticmethod
    def preprocess_data(X, y):
        return X.reshape(len(X), 28 * 28), keras.utils.to_categorical(y, num_classes=10)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # Load the data
    (X_train, y_train), (X_valid, y_valid) = keras.datasets.mnist.load_data()

    # Input shape is 28*28 since that is the size of the flattened array for each image
    model = Model([28 * 28], 10)

    # Fit the model to the data and plot the loss and validation loss
    model.fit(X_train, y_train, X_valid, y_valid)
    model.plot_iterations()