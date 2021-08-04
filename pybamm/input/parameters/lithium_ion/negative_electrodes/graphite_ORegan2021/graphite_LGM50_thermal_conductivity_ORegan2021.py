from pybamm import Parameter


def graphite_LGM50_thermal_conductivity_ORegan2021(T):
    """
    Wet negative electrode thermal conductivity as a function of the temperature from
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

    lambda_wet = -2.61e-4 * T ** 2 + 0.1726 * T - 24.49

    return lambda_wet
