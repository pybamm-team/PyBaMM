def electrolyte_conductivity_Ai2020(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration.
    Concentration should be in dm3 in the function.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
    167(1), 013512. DOI: 10.1149/2.0122001JES.
    .. [2] Torchio, Marcello, et al. "Lionsimba: a matlab framework based
    on a finite volume model suitable for li-ion battery design, simulation,
    and control." Journal of The Electrochemical Society 163.7 (2016): A1192.

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

    sigma_e = (
        1e-4
        * c_e
        * (
            (-10.5 + 0.668 * 1e-3 * c_e + 0.494 * 1e-6 * c_e ** 2)
            + (0.074 - 1.78 * 1e-5 * c_e - 8.86 * 1e-10 * c_e ** 2) * T
            + (-6.96 * 1e-5 + 2.8 * 1e-8 * c_e) * T ** 2
        )
        ** 2
    )

    return sigma_e
