import pybamm
from scipy import constants


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

    # The diffusivity epends on the electrolyte conductivity
    E_k_e = pybamm.Parameter("Electrolyte conductivity activation energy [J.mol-1]")
    inputs = {
        "Electrolyte concentration [mol.m-3]": c_e,
        "Temperature [K]": T,
        "Reference temperature [K]": T_inf,
        "Activation energy [J.mol-1]": E_k_e,
        "Ideal gas constant [J.mol-1.K-1]": R_g,
    }
    sigma_e = pybamm.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)

    # constants
    k_b = constants.physical_constants["Boltzmann constant"][0]
    F = constants.physical_constants["Faraday constant"][0]
    q_e = constants.physical_constants["electron volt"][0]

    D_c_e = (k_b / (F * q_e)) * sigma_e * T / c_e

    return D_c_e
