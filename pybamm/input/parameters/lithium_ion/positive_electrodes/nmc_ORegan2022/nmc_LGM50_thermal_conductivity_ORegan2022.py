def nmc_LGM50_thermal_conductivity_ORegan2022(T):
    """
    Wet positive electrode thermal conductivity as a function of the temperature from
    [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parameters of a high energy lithium-ion
    cylindrical battery." Electrochimica Acta 425 (2022): 140700

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
