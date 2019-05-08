#
# Equation classes for a Particle
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import autograd.numpy as np


class Standard(pybamm.SubModel):
    """Diffusion in the particles

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, c, j, broadcast=False):
        """
        PDE system for diffusion in the particles

        Parameters
        ----------
        c_e : :class:`pybamm.Variable`
            The particle concentration variable
        j : :class:`pybamm.Concatenation`
            Interfacial current density
        broadcast : bool
            Whether to broadcast variables when computing standard variables
        """
        param = self.set_of_parameters

        if len(c.domain) != 1:
            raise NotImplementedError(
                "Only implemented when c_k is on exactly 1 subdomain"
            )

        if c.domain[0] == "negative particle":
            N = -pybamm.grad(c)
            self.rhs = {c: -(1 / param.C_n) * pybamm.div(N)}
            self.algebraic = {}
            self.initial_conditions = {c: param.c_n_init}
            rbc = -param.C_n * j / param.a_n
            self.boundary_conditions = {
                c: {"left": (0, "Neumann"), "right": (rbc, "Neumann")}
            }
            self.variables = self.get_variables(c, N, broadcast)
        elif c.domain[0] == "positive particle":
            N = -pybamm.grad(c)
            self.rhs = {c: -(1 / param.C_p) * pybamm.div(N)}
            self.algebraic = {}
            self.initial_conditions = {c: param.c_p_init}
            rbc = -param.C_p * j / param.a_p / param.gamma_p
            self.boundary_conditions = {
                c: {"left": (0, "Neumann"), "right": (rbc, "Neumann")}
            }
            self.variables = self.get_variables(c, N, broadcast)
        else:
            raise pybamm.ModelError("Domain not valid for the particle equations")

        self.events = [pybamm.Function(np.min, c), pybamm.Function(np.max, c) - 1]

    def get_variables(self, c, N, broadcast):
        """
        Calculate dimensionless and dimensional variables for the electrolyte submodel

        Parameters
        ----------
        c : :class:`pybamm.Concatenation`
            The particle concentration variable
        N : :class:`pybamm.Symbol`
            The flux of lithium in the particles
        broadcast : bool
            Whether to broadcast variables when computing standard variables

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        if c.domain == ["negative particle"]:
            conc_scale = self.set_of_parameters.c_n_max
            domain = "Negative particle"
            broadcast_domain = ["negative electrode"]
        elif c.domain == ["positive particle"]:
            conc_scale = self.set_of_parameters.c_p_max
            domain = "Positive particle"
            broadcast_domain = ["positive electrode"]

        c_surf = pybamm.surf(c)
        if broadcast:
            c_surf = pybamm.Broadcast(c_surf, broadcast_domain)

        return {
            domain + " concentration": c,
            domain + " surface concentration": c_surf,
            domain + " flux": N,
            domain + " concentration [mols m-3]": conc_scale * c,
            domain + " surface concentration [mols m-3]": conc_scale * c_surf,
        }
