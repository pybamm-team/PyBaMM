"""
Adding particle-size distribution parameter values to a parameter set
"""


import pybamm
import numpy as np


def get_size_distribution_parameters(
    param,
    R_n_av=None,
    R_p_av=None,
    sd_n=0.3,
    sd_p=0.3,
    R_min_n=None,
    R_min_p=None,
    R_max_n=None,
    R_max_p=None,
):
    """
    A convenience method to add standard area-weighted particle-size distribution
    parameter values to a parameter set. A lognormal distribution is assumed for
    each electrode, with mean set equal to the particle radius parameter in the
    set (default) or a custom value. The standard deviations and min/max radii
    are specified relative (i.e. scaled by) the mean radius for convenience.
    Only the dimensional values are output from this method.

    Parameters
    ----------
    param : :class:`pybamm.ParameterValues`
        The parameter values to add the distribution parameters to.
    R_n_av : float (optional)
        The area-weighted mean particle radius (dimensional) of the negative electrode.
        Default is the value "Negative particle radius [m]" from param.
    R_p_av : float (optional)
        The area-weighted mean particle radius (dimensional) of the positive electrode.
        Default is the value "Positive particle radius [m]" from param.
    sd_n : float (optional)
        The area-weighted standard deviation, scaled by the mean radius R_n_av,
        hence dimensionless. Default is 0.3.
    sd_p : float (optional)
        The area-weighted standard deviation, scaled by the mean radius R_p_av,
        hence dimensionless. Default is 0.3.
    R_min_n : float (optional)
        Minimum radius in negative electrode, scaled by the mean radius R_n_av.
        Default is 0 or 5 standard deviations below the mean (if positive).
    R_min_p : float (optional)
        Minimum radius in positive electrode, scaled by the mean radius R_p_av.
        Default is 0 or 5 standard deviations below the mean (if positive).
    R_max_n : float (optional)
        Maximum radius in negative electrode, scaled by the mean radius R_n_av.
        Default is 5 standard deviations above the mean.
    R_max_p : float (optional)
        Maximum radius in positive electrode, scaled by the mean radius R_p_av.
        Default is 5 standard deviations above the mean.
    """

    # Radii from given parameter set
    R_n_typ = param["Negative particle radius [m]"]
    R_p_typ = param["Positive particle radius [m]"]

    # Set the mean particle radii for each electrode
    R_n_av = R_n_av or R_n_typ
    R_p_av = R_p_av or R_p_typ

    # Minimum radii
    R_min_n = R_min_n or np.max([0, 1 - sd_n * 5])
    R_min_p = R_min_p or np.max([0, 1 - sd_p * 5])

    # Max radii
    R_max_n = R_max_n or (1 + sd_n * 5)
    R_max_p = R_max_p or (1 + sd_p * 5)

    # Area-weighted particle-size distributions
    def f_a_dist_n_dim(R):
        return lognormal(R, R_n_av, sd_n * R_n_av)

    def f_a_dist_p_dim(R):
        return lognormal(R, R_p_av, sd_p * R_p_av)

    param.update(
        {
            "Negative area-weighted mean particle radius [m]": R_n_av,
            "Positive area-weighted mean particle radius [m]": R_p_av,
            "Negative area-weighted particle-size "
            + "standard deviation [m]": sd_n * R_n_av,
            "Positive area-weighted particle-size "
            + "standard deviation [m]": sd_p * R_p_av,
            "Negative minimum particle radius [m]": R_min_n * R_n_av,
            "Positive minimum particle radius [m]": R_min_p * R_p_av,
            "Negative maximum particle radius [m]": R_max_n * R_n_av,
            "Positive maximum particle radius [m]": R_max_p * R_p_av,
            "Negative area-weighted "
            + "particle-size distribution [m-1]": f_a_dist_n_dim,
            "Positive area-weighted "
            + "particle-size distribution [m-1]": f_a_dist_p_dim,
        },
        check_already_exists=False,
    )
    return param


def lognormal(x, x_av, sd):
    """
    A PyBaMM lognormal distribution for use with particle-size distribution models.
    The independent variable is x, range 0 < x < Inf, with mean x_av and standard
    deviation sd. Note: if, e.g. X is lognormally distributed, then the mean and
    standard deviations used here are of X rather than the normal distribution log(X).
    """

    mu_ln = pybamm.log(x_av**2 / pybamm.sqrt(x_av**2 + sd**2))
    sigma_ln = pybamm.sqrt(pybamm.log(1 + sd**2 / x_av**2))

    out = (
        pybamm.exp(-((pybamm.log(x) - mu_ln) ** 2) / (2 * sigma_ln**2))
        / pybamm.sqrt(2 * np.pi * sigma_ln**2)
        / x
    )
    return out
