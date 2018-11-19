#
# Equations for the electrode-electrolyte interface
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


class Interface(object):
    """Equations for the electrode-electrolyte interface."""

    def set_simulation(self, param, mesh):
        """
        Assign simulation-specific objects as attributes.

        Parameters
        ----------
        param : :class:`pybamm.parameters.Parameters` instance
            The parameters of the simulation
        operators : :class:`pybamm.operators.Operators` instance
            The spatial operators.
        mesh : :class:`pybamm.mesh.Mesh` instance
            The spatial and temporal discretisation.
        """
        self.param = param
        self.mesh = mesh

    def butler_volmer(self, domain, c=None, e=None):
        """Calculates the interfacial current densities
        using Butler-Volmer kinetics.

        Parameters
        ----------
        domain : string
            The domain in which to calculate the interfacial current density.
                * "xcn" or "xcp" : the interfacial current density is be
                    calculated in the relevant electrode as a function of c and
                    e. c and e should have the same shape as the relevant
                    electrode mesh.
                * "xcs" : the interfacial current density is zero
                * "xc" : the interfacial current density will be calculated
                    across the whole cell, calling on "xcn", "xcs" and "xcp".
                    c should have the same shape as self.mesh.xc.
        c : array_like, shape (n,), optional
            The electrolyte concentration.
        e : array_like, shape (n,), optional
            The potential difference.

        Returns
        -------
        j : array_like, shape (n,)
            The interfacial current density.

        """
        if domain == "xcn":
            assert c.shape == self.mesh.xcn.shape
            assert e.shape == self.mesh.xcn.shape
            j = self.param.iota_ref_n * c * np.sinh(e - self.param.U_Pb(c))
        elif domain == "xcs":
            j = np.zeros_like(self.mesh.xcs)
        elif domain == "xcp":
            assert c.shape == self.mesh.xcp.shape
            assert e.shape == self.mesh.xcp.shape
            j = (
                self.param.iota_ref_p
                * c ** 2
                * self.param.cw(c)
                * np.sinh(e - self.param.U_PbO2(c))
            )
        elif domain == "xc":
            assert c.shape == self.mesh.xc.shape
            assert e.shape[0] == len(self.mesh.xcn) + len(self.mesh.xcp)
            cn, cs, cp = np.split(
                c, np.array([self.mesh.nn - 1, self.mesh.nn + self.mesh.ns])
            )
            en, ep = np.split(e, np.array([self.mesh.nn - 1]))
            j = np.concatenate(
                [
                    self.butler_volmer(domain, c, e)
                    for c, e, domain in zip(
                        [cn, cs, cp], [en, None, ep], ["xcn", "xcs", "xcp"]
                    )
                ]
            )

        return j

    def uniform_current_density(self, domain, t):
        """Calculates the interfacial current densities
        using Butler-Volmer kinetics.

        Parameters
        ----------
        domain : string
            The domain in which to calculate the interfacial current density.
        t : float or array_like
            The time at which to evaluate the current density.

        Returns
        -------
        j : array_like
            The interfacial current density.

        """
        mesh = self.mesh

        if domain == "xcn":
            j = self.param.icell(t) / self.param.ln * np.ones_like(mesh.xcn)
        elif domain == "xcs":
            j = np.zeros_like(mesh.xcs)
        elif domain == "xcp":
            j = -self.param.icell(t) / self.param.lp * np.ones_like(mesh.xcp)
        elif domain == "xc":
            j = np.concatenate(
                [
                    self.uniform_current_density(domain, t)
                    for domain in ["xcn", "xcs", "xcp"]
                ]
            )

        return j
