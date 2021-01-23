from pybamm import exp, constants


def nca_diffusivity_Kim2011(sto, T):
    """
    NCA diffusivity as a function of stochiometry [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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

    D_ref = 3 * 10 ** (-15)
    E_D_s = 2e4
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
