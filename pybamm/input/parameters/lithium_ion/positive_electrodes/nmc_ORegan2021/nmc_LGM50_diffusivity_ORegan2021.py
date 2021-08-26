from pybamm import exp, constants


def nmc_LGM50_diffusivity_ORegan2021(sto, T):
    """
    NMC diffusivity as a function of stoichiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from [1].

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
    T: :class:`pybamm.Symbol`
       Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
       Solid diffusivity
    """

    a1 = -0.9231
    a2 = -0.4066
    a3 = -0.993
    b1 = 0.3216
    b2 = 0.4532
    b3 = 0.8098
    c0 = -13.96
    c1 = 0.002534
    c2 = 0.003926
    c3 = 0.09924
    d = 1449

    D_ref = (
        10
        ** (
            c0
            + a1 * exp(-((sto - b1) ** 2) / c1)
            + a2 * exp(-((sto - b2) ** 2) / c2)
            + a3 * exp(-((sto - b3) ** 2) / c3)
        )
        * 2.7  # correcting factor (see O'Regan et al 2021)
    )

    E_D_s = d * constants.R
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
