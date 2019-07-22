import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import checkersBoard


class ResNN():
    def __init__(self, lr=0.000001, residual_blocks=2, width=64):
        conv2d = keras.layers.Conv2D
        self.dformat = 'channels_first'
        board_height = checkersBoard.CheckersBoard.board_height
        board_width = checkersBoard.CheckersBoard.board_width

        self.board = keras.layers.Input(dtype=tf.float32, shape=[5, board_height, board_width])
        conv1 = keras.layers.BatchNormalization(axis=1)(
            conv2d(width, kernel_size=[3, 3], activation=tf.nn.relu, data_format=self.dformat, padding='same', use_bias=False)(
                self.board))
        res_tower = self.residual_block(conv1, width)
        for _ in range(residual_blocks - 1):
            res_tower = self.residual_block(res_tower, width)
        conv2 = keras.layers.BatchNormalization(axis=1)(
            conv2d(32, kernel_size=(1, 1), activation=tf.nn.relu, data_format=self.dformat, padding='same', use_bias=False)(
                res_tower))
        conv_flat = keras.layers.Flatten()(conv2)
        fc1 = keras.layers.Dense(width, activation=tf.nn.relu)(conv_flat)
        self.value = keras.layers.Dense(1, activation=tf.nn.tanh)(fc1)

        policy = keras.layers.BatchNormalization(axis=1)(conv2d(32, [1, 1], data_format=self.dformat, padding='same', use_bias=False)(res_tower))
        policy = keras.layers.Flatten()(policy)
        self.probabilities = keras.layers.Dense(checkersBoard.CheckersBoard.action_size, activation=keras.activations.softmax)(policy)

        self.model = keras.Model(inputs=self.board, outputs=[self.value, self.probabilities])
        self.model.compile(keras.optimizers.Adam(lr=lr), loss=[keras.losses.mean_squared_error, keras.losses.categorical_crossentropy])
        # self.model = keras.Model(inputs=self.board, outputs=[self.value])
        # self.model.compile(keras.optimizers.Adam(lr=lr),loss=[keras.losses.mean_squared_error])
        self.model.summary()

    def residual_block(self, input_layer, width):
        shortcut = input_layer

        residual = keras.layers.Conv2D(width, kernel_size=(3, 3), data_format=self.dformat, padding='same', use_bias=False)(input_layer)
        residual = keras.layers.BatchNormalization(axis=1)(residual)
        residual = keras.layers.ReLU()(residual)
        residual = keras.layers.Conv2D(width, kernel_size=(3, 3), data_format=self.dformat, padding='same', use_bias=False)(residual)
        residual = keras.layers.BatchNormalization(axis=1)(residual)
        add_shortcut = keras.layers.add([residual, shortcut])
        residual_result = keras.layers.ReLU()(add_shortcut)

        return residual_result

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
