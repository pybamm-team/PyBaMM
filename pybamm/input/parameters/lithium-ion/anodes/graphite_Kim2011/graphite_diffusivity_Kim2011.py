from pybamm import exp, constants, Scalar


def graphite_diffusivity_Kim2011(sto, T):
    """
    Graphite diffusivity [1].

    References
    ----------
    .. [1] Kim, G. H., Smith, K., Lee, K. J., Santhanagopalan, S., & Pesaran, A.
    (2011). Multi-domain modeling of lithium-ion batteries encompassing
    multi-physics in varied length scales. Journal of The Electrochemical
    Society, 158(8), A955-A969.

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

    D_ref = Scalar(9 * 10 ** (-14), "[m2.s-1]")
    E_D_s = Scalar(4e3, "[J.mol-1]")
    arrhenius = exp(E_D_s / constants.R * (1 / Scalar(298.15, "[K]") - 1 / T))

    return D_ref * arrhenius
