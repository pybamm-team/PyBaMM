from pybamm import exp


def electrolyte_conductivity_Nyman2008(c_e, T, T_inf, E_k_e, R_g):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].
    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.
    Parameters
    ----------
    c_e: :class: `numpy.Array`
        Dimensional electrolyte concentration
    T: :class: `numpy.Array`
        Dimensional temperature
    T_inf: double
        Reference temperature
    E_k_e: double
        Electrolyte conductivity activation energy
    R_g: double
        The ideal gas constant
    Returns
    -------
    :`numpy.Array`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3
        - 2.51 * (c_e / 1000) ** 1.5
        + 3.329 * (c_e / 1000)
    )

    arrhenius = exp(E_k_e / R_g * (1 / T_inf - 1 / T))

    return sigma_e * arrhenius
