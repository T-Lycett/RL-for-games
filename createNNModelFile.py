import cnn
import resNN


def create_model_file(filename):
    model = resNN.ResNN()
    model.save_model(filename)


if __name__ == '__main__':
    create_model_file('res128x5.h5')
