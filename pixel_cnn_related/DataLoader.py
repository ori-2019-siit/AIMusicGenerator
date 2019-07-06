import os
import numpy as np
from imageio import imread


def load(path, subset):  # path = folder sa slikama, subset = train|test, napravi svoju funkciju
    X = []
    for root, dirs, files in os.walk(os.path.join(path, subset)):
        for file in files:
            X.append(imread(os.path.join(root, file)).reshape((1, 64, 64, 3)))
    X = np.concatenate(X, axis=0)
    return X


class DataLoader(object):
    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data = load(data_dir, subset=subset)

        self.p = 0  # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset()  # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p: self.p + n]
        self.p += self.batch_size

        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
