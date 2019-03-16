import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import checkersBoard


class ResNN():
    def __init__(self, lr=0.001, residual_blocks=5):
        conv2d = keras.layers.Conv2D
        self.dformat = 'channels_first'
        board_height = checkersBoard.CheckersBoard.board_height
        board_width = checkersBoard.CheckersBoard.board_width

        self.board = keras.layers.Input(dtype=tf.float32, shape=[5, board_height, board_width])
        conv1 = keras.layers.BatchNormalization(axis=1)(
            conv2d(128, kernel_size=[3, 3], activation=tf.nn.relu, data_format=self.dformat, padding='same', use_bias=False)(
                self.board))
        res_tower = self.residual_block(conv1)
        for _ in range(residual_blocks - 1):
            res_tower = self.residual_block(res_tower)
        conv2 = keras.layers.BatchNormalization(axis=1)(
            conv2d(1, kernel_size=(1, 1), activation=tf.nn.relu, data_format=self.dformat, padding='same', use_bias=False)(
                res_tower))
        conv_flat = keras.layers.Flatten()(conv2)
        fc1 = keras.layers.Dense(128, activation=tf.nn.relu)(conv_flat)
        self.value = keras.layers.Dense(1, activation=tf.nn.tanh)(fc1)

        self.model = keras.Model(inputs=self.board, outputs=self.value)
        self.model.compile(keras.optimizers.Adam(lr=lr), loss=keras.losses.mean_squared_error)
        self.model.summary()

    def residual_block(self, input_layer):
        shortcut = input_layer

        residual = keras.layers.Conv2D(128, kernel_size=(3, 3), data_format=self.dformat, padding='same', use_bias=False)(input_layer)
        residual = keras.layers.BatchNormalization(axis=1)(residual)
        residual = keras.layers.ReLU()(residual)
        residual = keras.layers.Conv2D(128, kernel_size=(3, 3), data_format=self.dformat, padding='same', use_bias=False)(residual)
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

    def viewLayers(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            x = l.get_weights()
            print('LAYER ' + str(i))

            try:
                weights = x[0]
                s = weights.shape

                fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
                channel = 0
                filter = 0
                for i in range(s[2] * s[3]):
                    sub = fig.add_subplot(s[3], s[2], i + 1)
                    sub.imshow(weights[:, :, channel, filter], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    channel = (channel + 1) % s[2]
                    filter = (filter + 1) % s[3]

            except:

                try:
                    fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
                    for i in range(len(x)):
                        sub = fig.add_subplot(len(x), 1, i + 1)
                        if i == 0:
                            clim = (0, 2)
                        else:
                            clim = (0, 2)
                        sub.imshow([x[i]], cmap='coolwarm', clim=clim, aspect="auto")

                    plt.show()

                except:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")

                        plt.show()

                    except:
                        pass

            plt.show()
