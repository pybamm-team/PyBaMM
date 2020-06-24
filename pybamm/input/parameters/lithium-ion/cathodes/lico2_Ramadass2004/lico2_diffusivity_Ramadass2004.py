from pybamm import exp, constants


def lico2_diffusivity_Ramadass2004(sto, T):
    """
    LiCo2 diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Ramadass 2004.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

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
    D_ref = 1 * 10 ** (-14)
    E_D_s = 18550
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius
