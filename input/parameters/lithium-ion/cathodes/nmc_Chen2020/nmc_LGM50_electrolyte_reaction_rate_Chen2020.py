from pybamm import exp


def nmc_LGM50_electrolyte_reaction_rate_Chen2020(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between NMC and LiPF6 in EC:DMC.
    References
    ----------
    .. [1]
    Parameters
    ----------
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_r: double
        Reaction activation energy
    R_g: double
        The ideal gas constant
    Returns
    -------
    : double
        Reaction rate
    """
    m_ref = 3.59E-6
    arrhenius = exp(E_r / R_g * (1 / T_inf - 1 / T))

    return m_ref * arrhenius
