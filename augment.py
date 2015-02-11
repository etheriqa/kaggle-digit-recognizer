import numpy as np


class Augmenter(object):
    def __init__(self, shape, offset, length, slide):
        for i in range(2):
            assert shape[i] > 0
            assert offset[i] >= 0
            assert length[i] > 0
            assert slide[i] >= 0
            assert shape[i] >= offset[i] * 2 + length[i]
            assert slide[i] == 0 or (shape[i] - offset[i] * 2 - length[i]) % slide[i] == 0
            assert (shape[i] - offset[i] * 2 - length[i]) % 2 == 0

        self.__shape = shape
        self.__offset = offset
        self.__length = length
        self.__slide = slide
        self.__step = (
            int((shape[0] - offset[0] * 2 - length[0]) / slide[0]) if slide[0] > 0 else 1,
            int((shape[1] - offset[1] * 2 - length[1]) / slide[1]) if slide[1] > 0 else 1,
            )

    def factor(self):
        return (self.__slide[0] + 1) * (self.__slide[1] + 1)

    def dimension(self):
        return self.__length[0] * self.__length[1]

    def augment_training_data(self, training_data):
        augmented = np.empty((
            training_data.shape[0] * self.factor(),
            self.dimension()
            ))
        offsets = lambda i: range(
            self.__offset[i],
            self.__offset[i] + self.__step[i] * (self.__slide[i] + 1),
            self.__step[i]
            )
        index = 0
        for i in range(training_data.shape[0]):
            image = training_data[i].reshape(self.__shape)
            for x in offsets(0):
                for y in offsets(1):
                    augmented[index] = self.__trim(image, (x, y)).flatten()
                    index += 1
        return augmented


    def augment_training_target(self, training_target):
        augmented = np.empty((training_target.shape[0] * self.factor(),))
        index = 0
        for i in range(training_target.shape[0]):
            for j in range(self.factor()):
                augmented[index] = training_target[i]
                index += 1
        return augmented

    def augment_test_data(self, test_data):
        augmented = np.empty((
            test_data.shape[0],
            self.dimension()
            ))
        for i in range(test_data.shape[0]):
            image = test_data[i].reshape(self.__shape)
            x = self.__offset[0] + self.__step[0] * self.__slide[0] / 2
            y = self.__offset[1] + self.__step[1] * self.__slide[1] / 2
            augmented[i] = self.__trim(image, (x, y)).flatten()
        return augmented

    def __trim(self, image, offset):
        xs = slice(offset[0], offset[0] + self.__length[0])
        ys = slice(offset[1], offset[1] + self.__length[1])
        return image[xs, ys]


class AugmentedTrainingDataset(object):
    def __init__(self, dataset, augmenter):
        self.__dataset = dataset
        self.__augmenter = augmenter

    def size(self):
        return self.__dataset.size()

    def training_data(self, indices=None):
        return self.__augmenter.augment_training_data(self.__dataset.training_data(indices))

    def training_target(self, indices=None):
        return self.__augmenter.augment_training_target(self.__dataset.training_target(indices))

    def test_data(self, indices=None):
        return self.__augmenter.augment_test_data(self.__dataset.test_data(indices))

    def test_target(self, indices=None):
        return self.__dataset.test_target(indices)


class AugmentedTestDataset(object):
    def __init__(self, dataset, augmenter):
        self.__dataset = dataset
        self.__augmenter = augmenter

    def size(self):
        pass

    def test_data(self, indices=None):
        return self.__augmenter.augment_test_data(self.__dataset.test_data(indices))
