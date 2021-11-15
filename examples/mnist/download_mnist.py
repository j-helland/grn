import argparse

import tensorflow as tf
import torchvision as tv


def download_tf():
    tf.keras.datasets.mnist.load_data()

def download_torch():
    tv.datasets.MNIST(root='./', download=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./')
    args = parser.parse_args()

    download_tf()
    download_torch()
