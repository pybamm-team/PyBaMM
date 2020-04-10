from pybamm import exp, constants


def nca_electrolyte_reaction_rate_Kim2011(T):
    """
    Reaction rate for Butler-Volmer reactions between NCA and LiPF6 in EC:DMC
    [1].

     References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """
    i0_ref = 4  # reference exchange current density at 100% SOC
    sto = 0.41  # stochiometry at 100% SOC
    c_s_max = 4.9 * 10 ** 4  # max electrode concentration
    c_s_ref = sto * c_s_max  # reference electrode concentration
    c_e_ref = 1.2 * 10 ** 3  # reference electrolyte concentration
    alpha = 0.5  # charge transfer coefficient

    m_ref = (
        2
        * i0_ref
        / (c_e_ref ** alpha * (c_s_max - c_s_ref) ** alpha * c_s_ref ** alpha)
    )
    E_r = 3e4
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius
