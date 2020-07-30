import pybamm
import numpy as np


def graphite_lognormal_particle_size_distribution(R):
    """
    A lognormal particle-size distribution as a function of particle radius R. The mean
    of the distribution is equal to the "Partice radius [m]" from the parameter set,
    and the standard deviation is 0.3 times the mean.

    Parameters
    ----------
    R : :class:`pybamm.Symbol`
       Particle radius [m]

    """
    # Mean radius (dimensional)
    R_av = 1E-5

    # Standard deviation (dimensional)
    sd = R_av * 0.3

    # calculate usual lognormal parameters
    mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
    sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
    return (
        pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
        / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
        / (R)
    )
