from pybamm import exp, constants


def electrolyte_diffusivity_Ramadass2004(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration.

    References
    ----------
    .. [1] P. Ramadass, Bala Haran, Parthasarathy M. Gomadam, Ralph White, and Branko
    N. Popov. "Development of First Principles Capacity Fade Model for Li-Ion Cells."
    (2004)

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 7.5e-10
    E_D_e = 37040
    arrhenius = exp(E_D_e / constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius
