from pybamm import exp, constants, maximum


def nmc_LGM50_diffusivity_ORegan2021(sto, T):
    """
     NMC diffusivity as a function of stoichiometry, in this case the
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

    # a0 = -2.045
    # a1 = -1.132
    # a2 = -0.4122
    # a3 = -0.2109
    # b1 = 0.3193
    # b2 = 0.4542
    # b3 = 0.68
    # c0 = -13.3
    # c1 = 0.002554
    # c2 = 0.004368
    # c3 = 0.003742
    # d = 1607

    a1 = -0.9337
    a2 = -0.4581
    a3 = -0.9386
    b1 = 0.3222
    b2 = 0.4531
    b3 = 0.75
    c0 = -14
    c1 = 0.002572
    c2 = 0.004133
    c3 = 0.05436
    d = 1367

    # D_ref = 10 ** (
    #     a0 * sto
    #     + c0
    #     + a1 * exp(-((sto - b1) ** 2) / c1)
    #     + a2 * exp(-((sto - b2) ** 2) / c2)
    #     + a3 * exp(-((sto - b3) ** 2) / c3)
    # )

    D_ref = (
        10
        ** (
            c0
            + a1 * exp(-((sto - b1) ** 2) / c1)
            + a2 * exp(-((sto - b2) ** 2) / c2)
            + a3 * exp(-((sto - b3) ** 2) / c3)
        )
        * 2
    )

    # D_ref = 1e-14

    # D_ref = maximum(D_ref, 10 ** (-14.5))

    E_D_s = d * constants.R
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
