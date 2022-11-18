def electrolyte_diffusivity_Ai2020(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
    167(1), 013512. DOI: 10.1149/2.0122001JES.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration, mol/m^3
    T: :class:`pybamm.Symbol`
        Dimensional temperature, K


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 10 ** (-8.43 - 54 / (T - 229 - 5e-3 * c_e) - 0.22e-3 * c_e)

    return D_c_e
