import numpy as np


def multimutate(x, mu, sigma):
    nVar = len(x)
    
    # calc the number of elements to mutate
    nMu = int(np.ceil(mu * nVar))
    
    # select indices for mutation
    j = np.random.choice(nVar, nMu, replace=False)
    
    # Copy x to y to avoid modifying the original array
    y = np.copy(x)
    
    # apply mutations
    y[j] += sigma * np.random.randn(len(j))
    
    return y