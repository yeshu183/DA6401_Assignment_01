from keras.datasets import mnist, fashion_mnist
import numpy as np

def load_dataset(dataset_name):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    val_size = int(0.1 * len(x_train))
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_val = x_val.reshape(x_val.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train, y_train, x_val, y_val, x_test, y_test
