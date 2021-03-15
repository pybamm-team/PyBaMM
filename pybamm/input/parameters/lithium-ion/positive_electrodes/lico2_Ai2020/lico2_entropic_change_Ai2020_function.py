def lico2_entropic_change_Ai2020_function(sto):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Ref [1], which is only accurate
    for 0.43 < sto < 0.9936.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity
    in Lithium-Ion Pouch Cells. Journal of The Electrochemical Society,
        167(1), 013512. DOI: 10.1149/2.0122001JES

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """

    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    p1 = -3.20392657
    p2 = 14.5719049
    p3 = -27.9047599
    p4 = 29.1744564
    p5 = -17.992018
    p6 = 6.54799331
    p7 = -1.30382445
    p8 = 0.109667298

    du_dT = (
        p1 * sto ** 7
        + p2 * sto ** 6
        + p3 * sto ** 5
        + p4 * sto ** 4
        + p5 * sto ** 3
        + p6 * sto ** 2
        + p7 * sto
        + p8
    )

    return du_dT
