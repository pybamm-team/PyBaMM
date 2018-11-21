import numpy as np


def lco_ocp(sto):
    """
          Lithium Cobalt Oxide OCP taken from Newman's DUALFOIL code
          Measured by Oscar Garcia 2001 using Quallion electrodes for
          0.5 < sto < 0.99.  Fit revised by Karen Thomas in May 2003 to
          match Doyle's fit for sto < 0.4 and Garcia's data at larger y.
          Valid for 0 < sto < 0.99.

          Parameters
          ----------
          sto: double
               Stochiometry of material (li-fraction)

    """

    stretch = 1.062
    sto = stretch * sto

    u_eq = 2.16216 + 0.07645 * np.tanh(30.834 - 54.4806 * sto)
    +2.1581 * np.tanh(52.294 - 50.294 * sto)
    -0.14169 * np.tanh(11.0923 - 19.8543 * sto)
    +0.2051 * np.tanh(1.4684 - 5.4888 * sto)
    +0.2531 * np.tanh((-sto + 0.56478) / 0.1316)
    -0.02167 * np.tanh((sto - 0.525) / 0.006)

    return u_eq


def nmc_ocp(sto):
    """
          A different ocp function in this style

          Parameters
          ----------
          sto: double
               Stochiometry of material (li-fraction)

    """
    u_eq = 1
    return u_eq
