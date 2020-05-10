from pybamm import exp, constants, Scalar


def electrolyte_conductivity_Ecker2015(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration [1, 2, 3].

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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    # mol/m^3 to mol/l
    cm = c_e / Scalar(1e3, "[mol.m-3]")

    # value at T = 296K
    sigma_e_296 = (
        Scalar(0.2667, "[S.m-1]") * cm ** 3
        - Scalar(1.2983, "[S.m-1]") * cm ** 2
        + Scalar(1.7919, "[S.m-1]") * cm
        + Scalar(0.1726, "[S.m-1]")
    )

    # add temperature dependence
    E_k_e = 1.71e4
    C = 296 * exp(E_k_e / (constants.R * Scalar(296, "[K]")))
    sigma_e = C * sigma_e_296 * exp(-E_k_e / (constants.R * T)) / T

    return sigma_e
