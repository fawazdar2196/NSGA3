# scalarise.py

import numpy as np

def perform_scalarizing(z, params):
    """
    Perform scalarizing based on reference points.
    Args:
        z (np.ndarray): Translated objective functions (shape: nPop x nObj).
        params (dict): Dictionary containing normalization parameters and reference points.
    Returns:
        dict: Updated parameters after scalarization.
    """
    nPop, nObj = z.shape

    if len(params["smin"]) != 0:
        zmax = params["zmax"]
        smin = params["smin"]
    else:
        zmax = np.zeros((nObj, nObj))
        smin = np.full(nObj, np.inf)
    
    for j in range(nObj):
        w = get_scalarizing_vector(nObj, j)
        
        s = np.max(z / w, axis=1)  # Vectorized computation
        
        sminj = np.min(s)
        ind = np.argmin(s)
        
        if sminj < smin[j]:
            if ind >= nPop:
                logging.error(f"Reference index {ind} is out of bounds for z with shape {z.shape}")
                continue  # Skip updating zmax for this objective
            zmax[j, :] = z[ind, :]  # Assign the corresponding objective values
            smin[j] = sminj
    
    params["zmax"] = zmax
    params["smin"] = smin
    return params

def get_scalarizing_vector(nObj, j):
    """
    Generates a scalarizing vector for objective j.
    Args:
        nObj (int): Number of objectives.
        j (int): Objective index.
    Returns:
        np.ndarray: Scalarizing vector.
    """
    epsilon = 1e-10
    w = epsilon * np.ones(nObj)
    w[j] = 1
    return w
