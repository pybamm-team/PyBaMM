from pybamm import exp


def electrolyte_diffusivity_Kim2011(c_e, T, T_inf, E_D_e, R_g):
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
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_D_e: double
        Electrolyte diffusion activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    :`numpy.Array`
        Solid diffusivity
    """

    D_c_e = (
        5.84 * 10 ** (-7) * exp(-2870 / T) * (c_e / 1000) ** 2
        - 33.9 * 10 ** (-7) * exp(-2920 / T) * (c_e / 1000)
        + 129 * 10 ** (-7) * exp(-3200 / T)
    )

    return D_c_e
