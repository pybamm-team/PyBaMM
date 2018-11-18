#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Interface(object):
    """Equations for the electrode-electrolyte interface."""

    def set_simulation(self, param):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.Parameters` instance
            The parameters of the simulation
        """
        self.param = param

    def butler_volmer(self, c, e, domain):
        """Calculates the interfacial current densities
        using Butler-Volmer kinetics.

        Parameters
        ----------
        c : array_like, shape (n,)
            The electrolyte concentration.
        e : array_like, shape (n,)
            The potential difference.
        domain : string
            The domain in which to calculate the interfacial current density.

        Returns
        -------
        j : array_like, shape (n,)
            The interfacial current density.

        """
        if domain == "xcn":
            j = self.param.iota_ref_n * c * np.sinh(e - self.param.U_Pb(c))
        elif domain == "xcs":
            j = 0 * c
        elif domain == "xcp":
            j = (
                self.param.iota_ref_p
                * c ** 2
                * self.param.cw(c)
                * np.sinh(e - self.param.U_PbO2(c))
            )

        return j
