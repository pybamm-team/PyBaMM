def aluminium_heat_capacity_CRC(T):
    """
    Aluminium specific heat capacity as a function of the temperature from [1].

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
       Specific heat capacity
    """

    cp = 4.503e-6 * T ** 3 - 6.256e-3 * T ** 2 + 3.281 * T + 355.7

    return cp
