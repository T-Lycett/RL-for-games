import tensorflow as tf
from tensorflow import keras
import checkersBoard

class CNN():
    def __init__(self, lr=0.001):
        conv2d = keras.layers.Conv2D
        dformat = 'channels_first'
        board_width = checkersBoard.CheckersBoard.board_width

        self.board = keras.layers.Input(dtype=tf.float32, shape=[5, 8, board_width])
        conv1 = keras.layers.BatchNormalization(axis=1)(
            conv2d(64, kernel_size=[3, 3], activation=tf.nn.relu, data_format=dformat, padding='same', use_bias=False)(
                self.board))
        conv2 = keras.layers.BatchNormalization(axis=1)(conv2d(64, kernel_size=(3, 3), activation=tf.nn.relu, data_format=dformat, padding='same', use_bias=False)(conv1))
        conv3 = keras.layers.BatchNormalization(axis=1)(conv2d(1, kernel_size=(1, 1), activation=tf.nn.relu, data_format=dformat, padding='same', use_bias=False)(conv2))
        # conv3 = keras.layers.BatchNormalization(axis=1)(conv2d(128, kernel_size=(3, 3), activation=tf.nn.relu, data_format=dformat, padding='same', use_bias=False)(conv2))
        conv_flat = keras.layers.Flatten()(conv3)
        fc1 = keras.layers.Dense(64, activation=tf.nn.relu)(conv_flat)
        self.value = keras.layers.Dense(1, activation=tf.nn.tanh)(fc1)

        self.model = keras.Model(inputs=self.board, outputs=self.value)
        self.model.compile(keras.optimizers.Adam(lr=lr), loss=tf.losses.mean_squared_error)
        self.model.summary()

    def set_lr(self, lr):
        self.model.compile(keras.optimizers.Adam(lr=lr), loss=tf.losses.mean_squared_error)

    def fit_model(self, states, targets, batch_size, epochs):
        self.model.fit(states, targets, batch_size=batch_size, epochs=epochs)

    def predict(self, features):
        return self.model.predict(features)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
