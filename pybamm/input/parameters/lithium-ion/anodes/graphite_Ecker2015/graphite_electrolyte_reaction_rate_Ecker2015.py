from pybamm import exp
from scipy import constants


def graphite_electrolyte_reaction_rate_Ecker2015(T, T_inf, E_r, R_g):
    """
    Reaction rate for Butler-Volmer reactions between graphite and LiPF6 in EC:DMC.

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
    :`numpy.Array`
        Reaction rate
    """

    k_ref = 1.995 * 1e-10

    # multiply by Faraday's constant to get correct units
    F = constants.physical_constants["Faraday constant"][0]
    m_ref = F * k_ref

    arrhenius = exp(-E_r / (R_g * T)) * exp(E_r / (R_g * T_inf))

    return m_ref * arrhenius
