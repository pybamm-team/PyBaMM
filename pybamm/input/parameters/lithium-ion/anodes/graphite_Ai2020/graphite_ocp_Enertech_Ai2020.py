

def graphite_ocp_Enertech_Ai2020(sto):
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

    p1 =    -4880
    p2 =     2.33E4
    p3 =    -4.71E4
    p4 =     5.243E4
    p5 =    -3.501E4
    p6 =     1.434E4
    p7 =    -3531
    p8 =     491.8
    p9 =    -34.43
    p10 =      1.12
    u_eq = p1*sto**9 + p2*sto**8 + p3*sto**7 + p4*sto**6 + p5*sto**5 + p6*sto**4 + p7*sto**3 + p8*sto**2 + p9*sto + p10

    return u_eq
