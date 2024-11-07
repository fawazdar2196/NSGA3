import numpy as np




def dominates(x, y):
    """
    Check if solution x dominates solution y.
    
    :param x: First solution. Can be a list of objective values or an object with a "cost" attribute.
    :param y: Second solution. Can be a list of objective values or an object with a "cost" attribute.
    :return: True if x dominates y, False otherwise.
    """
    x = x["Cost"]

    y = y["Cost"]
    
    # import pdb; pdb.set_trace()
    # convert to numpy arrays for element-wise comparison
    x = np.array(x)
    y = np.array(y)
    
    return np.all(x <= y) and np.any(x < y)

def non_dominated_sorting(pop):
    nPop = len(pop)
    
    # init domination set and dominated count
    for individual in pop:
        individual["DominationSet"] = []
        individual["DominatedCount"] = 0
    
    F = [[]] # the list of fronts that shall be built
    
    # non-dominated sorting
    for i in range(nPop):
        for j in range(i+1, nPop):
            p = pop[i]
            q = pop[j]

            # import pdb; pdb.set_trace()
            
            if dominates(p, q):
                p["DominationSet"].append(j)
                q["DominatedCount"] += 1
            
            if dominates(q, p):
                q["DominationSet"].append(i)
                p["DominatedCount"] += 1
            
            pop[i] = p
            pop[j] = q
        
        if pop[i]["DominatedCount"] == 0:
            F[0].append(i)
            pop[i]["Rank"] = 1
    
    k = 0
    
    while True:
        Q = []
        
        for i in F[k]:
            p = pop[i]
            
            for j in p["DominationSet"]:
                q = pop[j]
                q["DominatedCount"] -= 1
                
                if q["DominatedCount"] == 0:
                    Q.append(j)
                    q["Rank"] = k + 1
                
                pop[j] = q
        
        if not Q:
            break
        
        F.append(Q)
        k += 1
    
    return pop, F


