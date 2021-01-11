def graphite_entropy_Enertech_Ai2020_function(sto):
    """
    Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. The fit is taken
    from Ref [1], which is only accurate
    for 0.43 < sto < 0.9936.

    References
    ----------
    .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
    Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch Cells. # noqa
    Journal of The Electrochemical Society, 167(1), 013512. DOI: 10.1149/2.0122001JES # noqa

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    Returns
    -------
    :class:`pybamm.Symbol`
        Entropic change [V.K-1]
    """

    du_dT = (
        0.001
        * (
            0.005269056
            + 3.299265709 * sto
            - 91.79325798 * sto ** 2
            + 1004.911008 * sto ** 3
            - 5812.278127 * sto ** 4
            + 19329.7549 * sto ** 5
            - 37147.8947 * sto ** 6
            + 38379.18127 * sto ** 7
            - 16515.05308 * sto ** 8
        )
        / (
            1
            - 48.09287227 * sto
            + 1017.234804 * sto ** 2
            - 10481.80419 * sto ** 3
            + 59431.3 * sto ** 4
            - 195881.6488 * sto ** 5
            + 374577.3152 * sto ** 6
            - 385821.1607 * sto ** 7
            + 165705.8597 * sto ** 8
        )
    )

    return du_dT
