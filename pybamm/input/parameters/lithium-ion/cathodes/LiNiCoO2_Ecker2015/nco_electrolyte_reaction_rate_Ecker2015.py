from pybamm import exp, constants


def nco_electrolyte_reaction_rate_Ecker2015(T):
    """
    Reaction rate for Butler-Volmer reactions between NCO and LiPF6 in EC:DMC [1, 2, 3].

    References
    ----------
       .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.
    .. [3] Richardson, Giles, et. al. "Generalised single particle models for
    high-rate operation of graded lithium-ion electrodes: Systematic derivation
    and validation." Electrochemica Acta 339 (2020): 135862

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature
    Returns
    -------
    :class:`pybamm.Symbol`
        Reaction rate
    """

    k_ref = 5.196e-11

    # multiply by Faraday's constant to get correct units
    m_ref = constants.F * k_ref

    E_r = 4.36e4
    arrhenius = exp(-E_r / (constants.R * T)) * exp(E_r / (constants.R * 296.15))

    return m_ref * arrhenius
