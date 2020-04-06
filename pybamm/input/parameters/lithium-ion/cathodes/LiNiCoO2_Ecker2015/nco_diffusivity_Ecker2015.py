from pybamm import exp


def nco_diffusivity_Ecker2015(sto, T, T_inf, E_D_s, R_g):
    """
    NCO diffusivity as a function of stochiometry [1, 2, 3].

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
    sto: :class: `numpy.Array`
        Electrode stochiometry
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_D_s: double
        Solid diffusion activation energy
    R_g: double
        The ideal gas constant

    Returns
    -------
    : double
        Solid diffusivity
    """

    D_ref = 3.7e-13 - 3.4e-13 * exp(-12 * (sto - 0.62) * (sto - 0.62))
    arrhenius = exp(-E_D_s / (R_g * T)) * exp(E_D_s / (R_g * T_inf))

    return D_ref * arrhenius
