from pybamm import exp, constants


def graphite_diffusivity_Ecker2015(sto, T):
    """
    Graphite diffusivity as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.

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

   ### Updated quintuple Gaussian fit to log10(D)
   
    y0 = -15.03
    A1 = 2.717
    sigma1 = 0.0791
    A2 = 1.471
    mu2 = 0.1715
    sigma2 = 0.1126
    A3 = 1.048
    sigma3 = 0.04042
    A4 = 1.514
    sigma4 = 0.04222
    A5 = 1.187
    sigma5 = 0.02648

    log_D_ref = y0 + (
        A1 * exp(-(sto / sigma1) ** 2) +
        A2 * exp(-((sto - mu2) / sigma2) ** 2) +
        A3 * exp(-((sto - 0.3118) / sigma3) ** 2) +
        A4 * exp(-((sto - 0.6169) / sigma4) ** 2) +
        A5 * exp(-((sto - 1) / sigma5) ** 2)
    )
    D_ref = 10 ** log_D_ref
    E_D_s = 3.03e4
    arrhenius = exp(-E_D_s / (constants.R * T)) * exp(E_D_s / (constants.R * 296))

    return D_ref * arrhenius
