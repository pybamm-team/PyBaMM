#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


def butler_volmer(param, cn, cs, cp, en, ep):
    """Calculates the interfacial current densities
    using Butler-Volmer kinetics.

    Parameters
    ----------
    param : pybamm.parameters.Parameters() instance
        The parameters of the simulation.
    cn : array_like, shape (n,)
        The electrolyte concentration in the negative electrode.
    cs : array_like, shape (s,)
        The electrolyte concentration in the separator.
    cp : array_like, shape (p,)
        The electrolyte concentration in the positive electrode.
    en : array_like, shape (n,)
        The potential difference in the negative electrode.
    ep : array_like, shape (p,)
        The potential difference in the positive electrode.

    Returns
    -------
    j : array_like, shape (n+s+p,)
        The interfacial current density across the whole cell.
    jn : array_like, shape (n,)
        The interfacial current density in the negative electrode.
    jp : array_like, shape (p,)
        The interfacial current density in the positive electrode.

    """
    jn = param.iota_ref_n * cn * np.sinh(en - param.U_Pb(cn))
    js = 0 * cs
    jp = (
        param.iota_ref_p
        * cp ** 2
        * param.cw(cp)
        * np.sinh(ep - param.U_PbO2(cp))
    )

    j = np.concatenate([jn, js, jp])
    return j, jn, jp
