from pybamm import exp, tanh


def graphite_ocp_Enertech_Ai2020_function(sto):
    """
    Graphite  Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
       for 0.0065 < sto < 0.84.

        References
        ----------
        .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020). 
        Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in Lithium-Ion Pouch Cells.
        Journal of The Electrochemical Society, 167(1), 013512. DOI: 10.1149/2.0122001JES

    Parameters
    ----------
    sto: double
       Stochiometry of material (li-fraction)

    """

    #  p1 = 0.07031
    #  p2 = -0.4612
    #  u_eq = p1 * sto ** p2

    u_eq = (
        0.194
        + 1.5 * exp(-120.0 * sto)
        + 0.0351 * tanh((sto - 0.286) / 0.083)
        - 0.0045 * tanh((sto - 0.849) / 0.119)
        - 0.035 * tanh((sto - 0.9233) / 0.05)
        - 0.0147 * tanh((sto - 0.5) / 0.034)
        - 0.102 * tanh((sto - 0.194) / 0.142)
        - 0.022 * tanh((sto - 0.9) / 0.0164)
        - 0.011 * tanh((sto - 0.124) / 0.0226)
        + 0.0155 * tanh((sto - 0.105) / 0.029)
    )
    #
    return u_eq
