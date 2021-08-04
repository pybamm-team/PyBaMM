def copper_heat_capacity_CRC(T):
    """
    Copper specific heat capacity as a function of the temperature from [1].

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

    cp = 1.445e-6 * T ** 3 - 1.946e-3 * T ** 2 + 0.9633 * T + 236

    return cp
