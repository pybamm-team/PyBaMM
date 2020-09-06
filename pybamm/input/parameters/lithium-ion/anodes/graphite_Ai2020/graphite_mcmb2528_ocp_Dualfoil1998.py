from pybamm import exp, tanh


def graphite_mcmb2528_ocp_Dualfoil1998(sto):
    """
       Graphite MCMB 2528 Open Circuit Potential (OCP) as a function of the
       stochiometry. The fit is taken from Dualfoil [1]. Dualfoil states that the data
       was measured by Chris Bogatu at Telcordia and PolyStor materials, 2000. However,
       we could not find any other records of this measurment.

       References
       ----------
       .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html
       """

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

    return u_eq
