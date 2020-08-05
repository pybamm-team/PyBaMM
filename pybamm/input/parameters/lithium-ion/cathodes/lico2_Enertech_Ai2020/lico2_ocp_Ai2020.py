import autograd.numpy as np


def lico2_ocp_Ai2020(sto):
    """
    Lithium Cobalt Oxide (LiCO2) Open Circuit Potential (OCP) as a a function of the
    stochiometry. The fit is taken from the Enertech cell [1], which is only accurate
       for 0.435 < sto < 0.9651.

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

    p1 =    -1.025E5
    p2 =     6.429E5
    p3 =    -1.777E6
    p4 =     2.838E6
    p5 =    -2.888E6
    p6 =     1.940E6
    p7 =    -8.609E5
    p8 =     2.432E5
    p9 =    -2.968E4
    p10 =      2855
    u_eq = p1*sto ** 9 + p2*sto ** 8 + p3*sto ** 7 + p4*sto ** 6 + p5*sto ** 5 + p6*sto ** 4 + p7*sto ** 3 + p8*sto ** 2 + p9*sto + p10

    return u_eq
