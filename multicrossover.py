import numpy as np

def multicrossover(x1, x2):

    # creates random numbers (between 0 and 1) in the same shape as x1
    alpha = np.random.rand(*x1.shape)

    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1

    return y1, y2

















