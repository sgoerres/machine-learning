import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expvals = np.exp(L)
    denom = np.sum(expvals)
    return np.divide(expvals, denom)

