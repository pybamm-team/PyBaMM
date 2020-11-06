from pybamm import Parameter


def dlnf_dlnc_Ai2020(c_e, T, T_ref=298.3, t_plus=0.38):
    """
    Activity dependence of LiPF6 in EC:DMC as a function of ion concentration.

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
        1 + dlnf/dlnc
    """
    T_ref = Parameter("Reference temperature [K]")
    t_plus = Parameter("Cation transference number")
    dlnf_dlnc = (
        0.601
        - 0.24 * (c_e / 1000) ** 0.5
        + 0.982 * (1 - 0.0052 * (T - T_ref)) * (c_e / 1000) ** 1.5
    ) / (1 - t_plus)
    return dlnf_dlnc
