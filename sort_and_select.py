import numpy as np
from normalise_pop import normalize_population
from nondom_sorting import non_dominated_sorting
from associate_to_reference import associate_to_reference_point

def sort_and_select_population(pop, params):
    pop, params = normalize_population(pop, params)
    pop, F = non_dominated_sorting(pop)
    
    nPop = params["nPop"]
    
    if len(pop) == nPop:
        return pop, F, params
    
    pop, d, rho = associate_to_reference_point(pop, params)
    
    newpop = []
    LastFront = []  # Initialize LastFront variable

    # Create a new population starting with the best performing individuals
    for l, front in enumerate(F):
        if len(newpop) + len(front) > nPop:
            LastFront = front  # Save the last front
            break
        newpop.extend([pop[i] for i in front])
    
    # Process the last front to fill up the population
    while True:
        if not LastFront:  # Prevent infinite loop if LastFront is empty
            break

        j = rho.argmin()
        
        # Get the members of LastFront associated with the least populated niche
        AssociatedFromLastFront = [i for i in LastFront if pop[i]["AssociatedRef"] == j]
        
        # If no members are associated with that niche, set to inf and continue
        if not AssociatedFromLastFront:
            rho[j] = float("inf")
            continue

        # Select a new member to add to newpop
        if rho[j] == 0:
            ddj = d[AssociatedFromLastFront, j]
            new_member_ind = ddj.argmin()
        else:
            new_member_ind = np.random.randint(len(AssociatedFromLastFront))
        
        MemberToAdd = AssociatedFromLastFront[new_member_ind]
        
        LastFront.remove(MemberToAdd)
        newpop.append(pop[MemberToAdd])
        
        rho[j] += 1
        
        if len(newpop) >= nPop:
            break
    
    # Final sorting of the new population
    pop, F = non_dominated_sorting(newpop)
    
    return pop, F, params
