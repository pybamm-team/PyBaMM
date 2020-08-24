

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

    p1 =  0.07031
    p2 =  -0.4612
    u_eq = p1 * sto ** p2
    # 
    return u_eq
