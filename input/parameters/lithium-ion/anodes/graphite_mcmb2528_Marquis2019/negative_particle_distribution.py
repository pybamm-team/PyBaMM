import pybamm 
import numpy as np

def negative_particle_distribution(x):
    
    return pybamm.Function(np.ones_like,x)
