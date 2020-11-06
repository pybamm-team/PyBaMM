def graphite_ocp_Enertech_Ai2020_function(sto):
    """
    Graphite  Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
       for 0.0065 < sto < 0.84.

        References
        ----------
        .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
        Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
        Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1),
        013512. DOI: 10.1149/2.0122001JES

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        OCP [V]

    """

    p1 = -2058.29865
    p2 = 10040.08960
    p3 = -20824.86740
    p4 = 23911.86578
    p5 = -16576.3692
    p6 = 7098.09151
    p7 = -1845.43634
    p8 = 275.31114
    p9 = -21.20097
    p10 = 0.84498
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
