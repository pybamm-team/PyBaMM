from pybamm import exp, constants


def nmc_LGM50_electronic_conductivity_ORegan2021(T):
    """
    Positive electrode electronic conductivity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] K. O'Regan ... (2021)

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    E_r = 3.5e3
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    sigma = 0.8473 * arrhenius

    return sigma
