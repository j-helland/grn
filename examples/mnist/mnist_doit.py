# ! /usr/bin/doit -f

from functools import partial
from pathlib import Path
from doit.task import clean_targets


def clean_glob_paths(paths: list):
    for path in paths:
        p = Path(path)
        for item in p.parent.glob(p.name):
            print(f'removing file {str(item)}')
            item.unlink()


"""
Tensorflow
"""
def task_0():
    return {
        'doc': 'Tensorflow MNIST training 0',
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'tf_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_1():
    return {
        'doc': 'Tensorflow MNIST training 1',
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'tf_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_2():
    return {
        'doc': 'Tensorflow MNIST training 2',
        'file_dep': [ ],
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
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'torch_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_4():
    return {
        'doc': 'PyTorch MNIST training 1',
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'torch_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_5():
    return {
        'doc': 'PyTorch MNIST training 2',
        'file_dep': [ ],
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
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_7():
    return {
        'doc': 'JAX FLAX MNIST training 1',
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }


def task_8():
    return {
        'doc': 'JAX FLAX MNIST training 2',
        'file_dep': [ ],
        'targets': [ ],
        'actions': [['python', 'jax_mnist_train.py']],
        'clean': [clean_targets],
    }