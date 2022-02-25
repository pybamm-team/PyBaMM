def lico2_ocp_Ai2020_function(sto):
    """
     Lithium Cobalt Oxide (LiCO2) Open Circuit Potential (OCP) as a a function of the
     stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
        for 0.435 < sto < 0.9651.

     References
     ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020). Electrochemical
     Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch Cells.
     Journal of The Electrochemical Society, 167(1), 013512. DOI: 10.1149/2.0122001JES

     Parameters
     ----------
     sto: double
        Stochiometry of material (li-fraction)

    Returns
     -------
     :class:`pybamm.Symbol`
         OCP [V]

    """

    p1 = -107897.40
    p2 = 677406.28
    p3 = -1873803.91
    p4 = 2996535.44
    p5 = -3052331.36
    p6 = 2053377.31
    p7 = -912135.88
    p8 = 257964.35
    p9 = -42146.98
    p10 = 3035.67
    u_eq = (
        p1 * sto ** 9
        + p2 * sto ** 8
        + p3 * sto ** 7
        + p4 * sto ** 6
        + p5 * sto ** 5
        + p6 * sto ** 4
        + p7 * sto ** 3
        + p8 * sto ** 2
        + p9 * sto
        + p10
    )
    return u_eq
