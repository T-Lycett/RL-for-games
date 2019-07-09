import cnn
import resNN


def create_model_file(filename):
    model = resNN.ResNN(lr=0.0001, residual_blocks=5, width=128)
    model.save_model(filename)


if __name__ == '__main__':
    create_model_file('res128x5_dual_lr10-3.h5')
