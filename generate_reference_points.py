import numpy as np

def generate_reference_points(M, p):
    """
    Generates reference points, uniformly distributed
    
    :param M: number of objectives.
    :param p: Division parameter.
    :return: A set of reference points as a numpy array.
    """
    Zr = get_fixed_row_sum_integer_matrix(M, p).T / p
    return Zr

def get_fixed_row_sum_integer_matrix(M, RowSum):
    if M < 1:
        raise ValueError("M cannot be less than 1.")
    
    if not float(M).is_integer():
        raise ValueError("M must be an integer.")
    
    if M == 1:
        return np.array([[RowSum]])
    
    A = np.empty((0, M), int)
    for i in range(RowSum + 1):
        B = get_fixed_row_sum_integer_matrix(M - 1, RowSum - i)
        # prepend column i to B and append to A
        A = np.vstack((A, np.hstack((i*np.ones((B.shape[0], 1), dtype=int), B))))
    
    return A