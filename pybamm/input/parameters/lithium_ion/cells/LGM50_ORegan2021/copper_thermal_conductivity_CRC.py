def copper_thermal_conductivity_CRC(T):
    """
    Copper thermal conductivity as a function of the temperature from [1].

    References
    ----------
    .. [1] William M. Haynes (Ed.). "CRC handbook of chemistry and physics". CRC Press
    (2014).

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Thermal conductivity
    """

    lambda_th = -5.409e-7 * T ** 3 + 7.054e-4 * T ** 2 - 0.3727 * T + 463.6

    return lambda_th
