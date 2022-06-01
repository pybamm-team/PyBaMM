def electrolyte_conductivity_Nyman2008(c_e, T):
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
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    if c_e < 2000:
        sigma_e = ( 0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000))
    else:
        sigma_e = ( 0.1297 * (2000 / 1000) ** 3 - 2.51 * (2000 / 1000) ** 1.5 + 3.329 * (2000 / 1000))
    # sigma_e = 0.1 * 0.06248 * (1+298.15-0.05559) * (c_e/1e3) * (1 - 3.084 *(c_e/1e3)**0.5 + 1.33 *(1+ 0.03633 *(exp(1000/298.15))*c_e/1e3)   ) / (1+(c_e/1e3)**4*( 0.00795 *exp(1000/298.15))) 

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e
