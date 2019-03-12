import cnn


def create_model_file(filename):
    model = cnn.CNN()
    model.save_model(filename)


if __name__ == '__main__':
    create_model_file('cnn128x3.h5')
