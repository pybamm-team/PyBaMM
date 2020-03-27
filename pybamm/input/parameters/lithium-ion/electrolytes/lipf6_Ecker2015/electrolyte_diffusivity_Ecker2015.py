import pybamm
from pybamm import exp


def electrolyte_diffusivity_Ecker2015(c_e, T, T_inf, E_D_e, R_g):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

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

    # Depends on electrolyte conductivity. Have just hard coded in now for
    # convinience, but should be able to call the conductivity directly

    # mol/m^3 to mol/l
    cm = 1e-3 * c_e

    # value at T = 296K
    sigma_e_296 = 0.2667 * cm ** 3 - 1.2983 * cm ** 2 + 1.7919 * cm + 0.1726

    # add temperature dependence
    C = 296 * exp(E_D_e / (R_g * 296))
    sigma_e = C * sigma_e_296 * exp(-E_D_e / (R_g * T)) / T

    ## Depends on the electrolyte conductivity
    # E_k_e = pybamm.Parameter("Electrolyte conductivity activation energy [J.mol-1]")
    # sigma_e = pybamm.FunctionParameter(
    #    "Electrolyte conductivity [S.m-1]", c_e, T, T_inf, E_k_e, R_g
    # )

    # constants
    k_b = 1.38 * 1e-23
    F = 96487
    q_e = 1.602 * 1e-19

    D_c_e = (k_b / (F * q_e)) * sigma_e * T / c_e

    return D_c_e
