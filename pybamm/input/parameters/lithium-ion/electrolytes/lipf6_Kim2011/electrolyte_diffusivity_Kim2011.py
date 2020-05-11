from pybamm import exp


def electrolyte_diffusivity_Kim2011(c_e, T):
    """
    Diffusivity of LiPF6 in EC as a function of ion concentration from [1].

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

    D_c_e = (
        5.84 * 10 ** (-7) * exp(-2870 / T) * (c_e / 1000) ** 2
        - 33.9 * 10 ** (-7) * exp(-2920 / T) * (c_e / 1000)
        + 129 * 10 ** (-7) * exp(-3200 / T)
    )

    return D_c_e
