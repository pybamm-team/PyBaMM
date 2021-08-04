from pybamm import exp, constants, maximum


def graphite_LGM50_diffusivity_ORegan2021(sto, T):
    """
    LG M50 Graphite diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
       Electrode stochiometry
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """

    #  a0 = 5
    #  a1 = -1.38
    #  a2 = -6.278
    #  a3 = -5.033
    #  a4 = 3.998
    #  b1 = 0.2059
    #  b2 = 0.5882
    #  b3 = 0.8757
    #  b4 = 0.5956
    #  c0 = -14.84
    #  c1 = 0.0006043
    #  c2 = 0.0448
    #  c3 = 0.02463
    #  c4 = 0.004477
    #  d = 3221

    #  a0 = 9.947
    #  a1 = -1.737
    #  #  a2 = -6.565
    #  a2 = -3.6
    #  #  a3 = -9.47
    #  a3 = -7.54
    #  #  a4 = 2.955
    #  a4 = 1.704
    #  b1 = 0.1981
    #  b2 = 0.5399
    #  b3 = 0.8868
    #  b4 = 0.5858
    #  c0 = -15.14
    #  c1 = 0.0005582
    #  c2 = 0.04973
    #  c3 = 0.0478
    #  c4 = 0.002958
    #  d = 3221

    a0 = 11.17
    a1 = -1.553
    a2 = -6.136
    a3 = -9.725
    a4 = 1.85
    b1 = 0.2031
    b2 = 0.5375
    b3 = 0.9144
    b4 = 0.5953
    c0 = -15.11
    c1 = 0.0006091
    c2 = 0.06438
    c3 = 0.0578
    c4 = 0.001356
    d = 2092

    D_ref = (
        10
        ** (
            a0 * sto
            + c0
            + a1 * exp(-((sto - b1) ** 2) / c1)
            + a2 * exp(-((sto - b2) ** 2) / c2)
            + a3 * exp(-((sto - b3) ** 2) / c3)
            + a4 * exp(-((sto - b4) ** 2) / c4)
        )
        * 3
    )

    E_D_s = d * constants.R
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
