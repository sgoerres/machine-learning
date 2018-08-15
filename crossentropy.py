import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    sum = 0
    for i in range(0,len(Y)):
        tmp1 = Y[i] * np.log(P[i])
        tmp2 = (1 - Y[i] * np.log(1 - P[i]))
        tmp = tmp1 + tmp2
        sum = sum - tmp
        print(tmp1 + " + " + tmp2 + " = " + tmp)
    return sum