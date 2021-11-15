# ! /usr/bin/doit -f

from functools import partial
from pathlib import Path
from doit.task import clean_targets


MNIST_FILES = ['/root/.keras/datasets/mnist.npz', './MNIST/raw/train-images-idx3-ubyte']


def clean_glob_paths(paths: list):
    for path in paths:
        p = Path(path)
        for item in p.parent.glob(p.name):
            print(f'removing file {str(item)}')
            item.unlink()


def task_download_mnist():
    return {
        'doc': 'Download torchvision and keras versions of MNIST.',
        'file_dep': [],
        'targets': MNIST_FILES,
        'actions': [['python', 'download_mnist.py']],
        'clean': [clean_targets],
    }


"""
Tensorflow
"""
def task_0():
    return {
        'doc': 'Tensorflow MNIST training 0',
        'file_dep': MNIST_FILES,
        'targets': [],
        'actions': [['python', 'tf_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_1():
    return {
        'doc': 'Tensorflow MNIST training 1',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'tf_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_2():
    return {
        'doc': 'Tensorflow MNIST training 2',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'tf_mnist_train.py']],
        'clean': [clean_targets],
    }


"""
PyTorch
"""
def task_3():
    return {
        'doc': 'PyTorch MNIST training 0',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'torch_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_4():
    return {
        'doc': 'PyTorch MNIST training 1',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'torch_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_5():
    return {
        'doc': 'PyTorch MNIST training 2',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'torch_mnist_train.py']],
        'clean': [clean_targets],
    }


"""
JAX 
"""
def task_6():
    return {
        'doc': 'JAX FLAX MNIST training 0',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_7():
    return {
        'doc': 'JAX FLAX MNIST training 1',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_8():
    return {
        'doc': 'JAX FLAX MNIST training 2',
        'file_dep': MNIST_FILES,
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }