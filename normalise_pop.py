# normalize_pop.py

import numpy as np
from update_ideal_point import update_ideal_point
from scalarise import perform_scalarizing
import logging

def find_hyperplane_intercepts(zmax):
    """
    Finds the intercepts of the hyperplane for normalization.
    Args:
        zmax (np.ndarray): zmax matrix (shape: nObj x nObj).
    Returns:
        np.ndarray: Hyperplane intercepts (shape: nObj,).
    """
    a = np.max(zmax, axis=1)  # Shape: (nObj,)
    a = np.where(a == 0, 1e-10, a)  # Replace zeros with epsilon
    return a

def normalize_population(pop, params):
    """
    Normalize the population based on zmin and zmax.
    Args:
        pop (list): List of individuals in the population.
        params (dict): Dictionary containing normalization parameters.
    Returns:
        tuple: Normalized population and updated parameters.
    """
    # Update zmin (ideal point)
    params["zmin"] = update_ideal_point(pop, params.get("zmin", np.array([])))
    
    # Compute translated objective functions (fp)
    fp = np.array([p["Cost"] for p in pop]) - params["zmin"]  # Shape: (nPop, nObj)
    logging.debug(f"Translated objective functions (fp): {fp}")
    
    # Perform scalarizing to update zmax and smin
    params = perform_scalarizing(fp, params)
    
    # Compute hyperplane intercepts
    a = find_hyperplane_intercepts(params["zmax"])  # Shape: (nObj,)
    logging.debug(f"Hyperplane intercepts (a): {a}")
    
    # Normalize the objective functions
    for i, p in enumerate(pop):
        if np.all(a != 0):
            p["NormalizedCost"] = fp[i] / a  # Element-wise division
        else:
            logging.error("Hyperplane intercept 'a' contains zero. Cannot perform normalization.")
            p["NormalizedCost"] = np.full(a.shape, np.inf)  # Assign infinity if normalization fails
        logging.debug(f"Individual {i} Normalized Cost: {p['NormalizedCost']}")
    
    return pop, params
