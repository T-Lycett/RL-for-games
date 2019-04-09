import numpy as np


# code source: https://github.com/sherjilozair/char-rnn-tensorflow/issues/1
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1) * s))
