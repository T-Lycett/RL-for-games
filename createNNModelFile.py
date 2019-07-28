import cnn
import resNN


def create_model_file(filename):
    model = resNN.ResNN(lr=0.00001, residual_blocks=3, width=64, q_learning_only=True)
    model.save_model(filename)


if __name__ == '__main__':
    create_model_file('res64x3.h5')
