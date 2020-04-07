from pybamm import exp


def electrolyte_conductivity_Kim2011(c_e, T):
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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        3.45 * exp(-798 / T) * (c_e / 1000) ** 3
        - 48.5 * exp(-1080 / T) * (c_e / 1000) ** 2
        + 244 * exp(-1440 / T) * (c_e / 1000)
    )

    return sigma_e
