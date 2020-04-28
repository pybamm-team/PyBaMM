#
# Base class for particle cracking model
# It simulates how much surface area is created by cracks during battery cycling
# For setting up now and to be finished later

import pybamm


class ParticleCracking(pybamm.BaseSubModel):
    """cracking behaviour in electrode particles.
    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    """
    
