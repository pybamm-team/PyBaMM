from pybamm import exp, standard_parameters_lithium_ion


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
    sto : :class:`pybamm.Symbol`
        Electrode stochiometry
    T : :class:`pybamm.Symbol`
        Dimensional temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity [m2.s-1]
    """
    param = standard_parameters_lithium_ion
    D_ref = 3 * 10 ** (-15)
    arrhenius = exp(param.E_D_s_p / param.R * (1 / param.T_ref - 1 / T))

    return D_ref * arrhenius
