import numpy as np
import pandas as pd

from augment import *


class DatasetFactory():
    def create_training_dataset(self, **kwargs):
        training_set = TrainingDataset(size=kwargs['training_size'])
        if kwargs['augment']:
            training_dataset = AugmentedTrainingDataset(
                training_set,
                self._augmenter(**kwargs)
            )
        return training_set

    def create_test_dataset(self, **kwargs):
        test_set = TestDataset(size=kwargs['test_size'])
        if kwargs['augment']:
            test_dataset = AugmentedTestDataset(
                test_set,
                self._augmenter(**kwargs)
            )
        return test_set

    def _augmenter(self, **kwargs):
        return Augmenter(
            (28, 28),
            (kwargs['augment_offset'], kwargs['augment_offset']),
            (kwargs['augment_length'], kwargs['augment_length']),
            (kwargs['augment_slide'], kwargs['augment_slide'])
        )


class TrainingDataset(object):
    def __init__(self, csv='data/train.csv', size=42000, shuffle=True):
        samples = pd.read_csv(csv).values
        if shuffle:
            np.random.shuffle(samples)
        self.__size = size
        self.__data = samples[:size, 1:].astype(float)
        self.__target = samples[:size, 0].astype(int)

    def size(self):
        return self.__size

    def training_data(self, indices=None):
        return self.test_data(indices)

    def training_target(self, indices=None):
        return self.test_target(indices)

    def test_data(self, indices=None):
        if indices is None:
            return self.__data
        else:
            return self.__data[indices]

    def test_target(self, indices=None):
        if indices is None:
            return self.__target
        else:
            return self.__target[indices]


class TestDataset(object):
    def __init__(self, csv='data/test.csv', size=28000):
        self.__size = size
        self.__data = pd.read_csv(csv).values.astype(float)[:size]

    def size(self):
        return self.__size

    def test_data(self, indices=None):
        if indices is None:
            return self.__data
        else:
            return self.__data[indices]


if __name__ == '__main__':
    training = TrainingDataset()
    assert training.size() == 42000
    assert training.training_data().shape == (42000, 28 ** 2)
    assert training.training_target().shape == (42000,)

    training = TrainingDataset(size=1000)
    assert training.size() == 1000
    assert training.training_data().shape == (1000, 28 ** 2)
    assert training.training_target().shape == (1000,)

    test = TestDataset()
    assert test.size() == 28000
    assert test.test_data().shape == (28000, 28 ** 2)

    test = TestDataset(size=1000)
    assert test.size() == 1000
    assert test.test_data().shape == (1000, 28 ** 2)
