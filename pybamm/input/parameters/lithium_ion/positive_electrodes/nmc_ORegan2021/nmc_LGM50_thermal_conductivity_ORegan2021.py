from pybamm import Parameter


def nmc_LGM50_thermal_conductivity_ORegan2021(T):
    """
    Wet positive electrode thermal conductivity as a function of the temperature from
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

    lambda_wet = 2.063e-5 * T ** 2 - 0.01127 * T + 2.331

    return lambda_wet
