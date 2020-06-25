from pybamm import exp, constants


def electrolyte_conductivity_Ramadass2004(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration.
    Concentration should be in dm3 in the function.

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
    # mol.m-3 to mol.dm-3, original function is likely in mS/cm
    # The function is not in Arora 2000 as reported in Ramadass 2004

    cm = 1e-6 * c_e  # here it should be only 1e-3

    sigma_e = (
        4.1253 * (10 ** (-4))
        + 5.007 * cm
        - 4.7212 * (10 ** 3) * (cm ** 2)
        + 1.5094 * (10 ** 6) * (cm ** 3)
        - 1.6018 * (10 ** 8) * (cm ** 4)
    ) * 1e3  # and here there should not be an exponent

    E_k_e = 34700
    arrhenius = exp(E_k_e / constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius
