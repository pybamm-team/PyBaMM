def graphite_entropic_change_PeymanMPM(sto):
    """
    Graphite entropic change in open circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from [1]

    References
    ----------
    .. [1] K.E. Thomas, J. Newman, "Heats of mixing and entropy in porous insertion
           electrode", J. of Power Sources 119 (2003) 844-849

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    du_dT = 10 ** (-3) * (
        0.28
        - 1.56 * sto
        - 8.92 * sto ** (2)
        + 57.21 * sto ** (3)
        - 110.7 * sto ** (4)
        + 90.71 * sto ** (5)
        - 27.14 * sto ** (6)
    )

    return du_dT
