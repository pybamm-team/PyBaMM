from pybamm import exp, tanh


def graphite_LGM50_entropic_change_ORegan2021(sto):
    """
    LG M50 Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry. The fit is taken from [1].

    References
    ----------
    .. [1] Kieran O’Regan, Ferran Brosa Planella, W. Dhammika Widanage, and Emma
    Kendrick. "Thermal-electrochemical parametrisation of a lithium-ion battery:
    mapping Li concentration and temperature dependencies." Journal of the
    Electrochemical Society, submitted (2021).

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Entropic change [V.K-1]
    """

    a0 = -0.1112
    a1 = -0.09002 * 0  # fixed fit (see discussion O'Regan et al 2021)
    a2 = 0.3561
    b1 = 0.4955
    b2 = 0.08309
    c0 = 0.02914
    c1 = 0.1122
    c2 = 0.004616
    d1 = 63.9

    dUdT = (
        a0 * sto
        + c0
        + a2 * exp(-((sto - b2) ** 2) / c2)
        + a1 * (tanh(d1 * (sto - (b1 - c1))) - tanh(d1 * (sto - (b1 + c1))))
    ) / 1000  # fit in mV / K

    return dUdT
