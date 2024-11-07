import numpy as np

def mutate(x, mu):
    nVar = len(x)
    nmu = int(np.ceil(mu * nVar))
    
    # randomly select indices for mutation
    j = np.random.choice(nVar, nmu, replace=False)
    y = np.copy(x)
    
    # flip the bits at the selected indices
    y[j] = 1 - y[j]
    return y

def multimutate(x, mu, sigma):
    nVar = len(x)
    y = np.copy(x)
    
    # Apply mutation based on the Gaussian noise model
    num_mutations = int(np.ceil(mu * nVar))
    indices = np.random.choice(nVar, num_mutations, replace=False)
    noise = np.random.normal(0, sigma, num_mutations)
    
    # Add the noise to the selected indices and clip to the valid range [0, 1] if needed
    y[indices] += noise
    y[indices] = np.clip(y[indices], 0, 1)  # Ensure the mutation stays within the bounds if required
    return y
