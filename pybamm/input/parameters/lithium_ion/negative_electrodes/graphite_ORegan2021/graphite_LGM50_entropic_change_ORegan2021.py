from pybamm import exp, tanh


def graphite_LGM50_entropic_change_ORegan2021(sto):
    """
    LG M50 Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry. The fit is taken from [1].

    References
    ----------
    .. [1] K. O'Regan ... (2021)

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
       Entropic change [V.K-1]
    """
    #  a0 = -0.09668
    #  a1 = -0.2434
    #  a2 = 0.3656
    #  b1 = 0.5203
    #  b2 = 0.08201
    #  c0 = 0.01831
    #  c1 = 0.007665
    #  c2 = 0.005254

    #  dUdT = (
    #      a0 * sto
    #      + c0
    #      + 0.5 * a1 * exp(-((sto - b1) ** 2) / c1)
    #      + a2 * exp(-((sto - b2) ** 2) / c2)
    #  ) / 1000  # fit in mV / K

    a0 = -0.1112
    a1 = -0.09002  # * 0
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
