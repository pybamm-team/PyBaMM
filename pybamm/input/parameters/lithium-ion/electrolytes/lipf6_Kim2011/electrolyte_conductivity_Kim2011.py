from pybamm import exp


def electrolyte_conductivity_Kim2011(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC as a function of ion concentration from [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

    Parameters
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_k_e: double
        Electrolyte conductivity activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Solid diffusivity
    """

    sigma_e = (
        3.45 * exp(-798 / T) * (c_e / 1000) ** 3
        - 48.5 * exp(-1080 / T) * (c_e / 1000) ** 2
        + 244 * exp(-1440 / T) * (c_e / 1000)
    )

    return sigma_e
