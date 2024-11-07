import numpy as np

def update_ideal_point(pop, prev_zmin=None):
    # the ideal point is the smallest objective value (currently found) in each dimension

    # initialise prev_zmin if it's None or empty
    if len(prev_zmin) == 0:
    
        # init prev_zmin with infinity with the size of the first individual's cost
        prev_zmin = np.full_like(pop[0]["Cost"], np.inf)
    
    zmin = prev_zmin
    for individual in pop:
        # update the ideal point (zmin) by taking the minimum between the current zmin and the individual's cost
        zmin = np.minimum(zmin, individual["Cost"])
    
    return zmin