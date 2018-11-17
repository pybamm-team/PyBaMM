import numpy as np


def graphite_ocp(sto):
    """
       Graphite OCP taken from Newman's DUALFOIL code (MCMB 2510 carbon (Bellcore))

       Parameters
       ----------
       sto: double
            Stochiometry of material (li-fraction)

       """

    u_eq = 0.194 + 1.5 * np.exp(-120.0 * sto)
    +0.0351 * np.tanh((sto - 0.286) / 0.083)
    - 0.0045 * np.tanh((sto - 0.849) / 0.119)
    - 0.035 * np.tanh((sto - 0.9233) / 0.05)
    - 0.0147 * np.tanh((sto - 0.5) / 0.034)
    - 0.102 * np.tanh((sto - 0.194) / 0.142)
    - 0.022 * np.tanh((sto - 0.9) / 0.0164)
    - 0.011 * np.tanh((sto - 0.124) / 0.0226)
    + 0.0155 * np.tanh((sto - 0.105) / 0.029)

    return u_eq


def silicon_ocp(sto):
    """
          A different ocp function in this style

          Parameters
          ----------
          sto: double
               Stochiometry of material (li-fraction)

    """
    u_eq = 1
    return u_eq
