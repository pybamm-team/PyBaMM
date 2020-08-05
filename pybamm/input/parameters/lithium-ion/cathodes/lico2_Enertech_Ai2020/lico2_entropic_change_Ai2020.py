import autograd.numpy as np


def lico2_entropic_change_Ai2020(sto, c_p_max):
    """
        Lithium Cobalt Oxide (LiCO2) entropic change in open circuit potential (OCP) at
        a temperature of 298.15K as a function of the stochiometry. The fit is taken
        from Ref [1], which is only accurate
       for 0.43 < sto < 0.9936.

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

    # Since the equation for LiCo2 from this ref. has the stretch factor,
    # should this too? If not, the "bumps" in the OCV don't line up.
    p1 =    -3204
    p2 =    1.457E4
    p3 =    -2.79E4
    p4 =     2.917E4
    p5 =    -1.799E4
    p6 =     6548
    p7 =     -1304
    p8 =     109.7 

    du_dT = p1*x**7 + p2*x**6 + p3*x**5 + p4*x**4 + p5*x**3 + p6*x**2 + p7*x + p8

    return du_dT
