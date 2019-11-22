import pybamm 
import numpy as np

def positive_particle_distribution(x):
    
    return pybamm.Function(np.ones_like,x)
