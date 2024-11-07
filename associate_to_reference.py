import numpy as np

def associate_to_reference_point(pop, params):
    Zr = params["Zr"]  
    nZr = params["nZr"]  
    
    rho = np.zeros(nZr) # rho counts the number of solutions associated to each niche/reference vector
    
    d = np.zeros((len(pop), nZr))

    for i, individual in enumerate(pop):
        for j in range(nZr):
            w = Zr[:, j] / np.linalg.norm(Zr[:, j]) 
            z = individual["NormalizedCost"]
            d[i, j] = np.linalg.norm(z - np.dot(w.T, z) * w) 
        
        dmin, jmin = d[i, :].min(), d[i, :].argmin()
        
        individual["AssociatedRef"] = jmin + 1 
        individual["DistanceToAssociatedRef"] = dmin
        rho[jmin] += 1  # increment count for the associated reference point/niche
    
    return pop, d, rho