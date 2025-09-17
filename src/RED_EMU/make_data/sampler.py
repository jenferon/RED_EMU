from scipy.stats import qmc

def LHS(lower_bounds, upper_bounds, dimensions, nitter):
    
    sampler = qmc.LatinHypercube(d=dimensions)
    sample = sampler.random(n=nitter)
    sample = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return sample