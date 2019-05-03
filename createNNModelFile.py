import cnn
import resNN


def create_model_file(filename):
    model = resNN.ResNN(residual_blocks=5, width=128)
    model.save_model(filename)


if __name__ == '__main__':
    create_model_file('res128x5T01.h5')
