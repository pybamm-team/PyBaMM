#
# Equation classes for the electrolyte concentration
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class StefanMaxwell(pybamm.SubModel):
    """"A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, c_e, reactions, epsilon=None):
        """
        PDE system for Stefan-Maxwell diffusion in the electrolyte

        Parameters
        ----------
        c_e : :class:`pybamm.Concatenation`
            Eletrolyte concentration
        reactions : dict
            Dictionary of reaction variables
        epsilon : :class:`pybamm.Symbol`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        """
        param = self.set_of_parameters

        # if porosity is not provided, use the input parameter
        if epsilon is not None:
            deps_dt = reactions["main"]["porosity change"]
        else:
            epsilon = param.epsilon
            deps_dt = pybamm.Scalar(0)

        # Flux
        N_e = -(epsilon ** param.b) * param.D_e(c_e) * pybamm.grad(c_e)

        # Model
        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]
        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)
        source_terms = param.s / param.gamma_e * j
        self.rhs = {
            c_e: (1 / epsilon)
            * (-pybamm.div(N_e) / param.C_e + source_terms - c_e * deps_dt)
        }

        self.initial_conditions = {c_e: param.c_e_init}
        self.boundary_conditions = {
            c_e: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        self.variables = self.get_variables(c_e, N_e)

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]

    def set_leading_order_system(self, c_e, reactions, epsilon=None):
        """
        ODE system for leading-order Stefan-Maxwell diffusion in the electrolyte
        Parameters
        ----------
        c_e : :class:`pybamm.Variable`
            Eletrolyte concentration
        reactions : dict
            Dictionary of reaction variables
        epsilon : :class:`pybamm.Concatenation`, optional
            Porosity. Default is None, in which case param.epsilon is used.
        """
        param = self.set_of_parameters

        # if porosity is not provided, use the input parameter
        if epsilon is not None:
            eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]
            deps_n_dt = sum(rxn["neg"]["deps_dt"] for rxn in reactions.values())
            deps_p_dt = sum(rxn["pos"]["deps_dt"] for rxn in reactions.values())
        else:
            eps_n = param.epsilon_n
            eps_s = param.epsilon_s
            eps_p = param.epsilon_p
            deps_n_dt = pybamm.Scalar(0)
            deps_p_dt = pybamm.Scalar(0)

        # Model
        source_terms = sum(
            param.l_n * rxn["neg"]["s_plus"] * rxn["neg"]["aj"]
            + param.l_p * rxn["pos"]["s_plus"] * rxn["pos"]["aj"]
            for rxn in reactions.values()
        )
        self.rhs = {
            c_e: 1
            / (param.l_n * eps_n + param.l_s * eps_s + param.l_p * eps_p)
            * (source_terms - c_e * (param.l_n * deps_n_dt + param.l_p * deps_p_dt))
        }
        self.initial_conditions = {c_e: param.c_e_init}

        # Variables
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        N_e = pybamm.Broadcast(0, whole_cell)
        c_e_var = pybamm.Concatenation(
            pybamm.Broadcast(c_e, ["negative electrode"]),
            pybamm.Broadcast(c_e, ["separator"]),
            pybamm.Broadcast(c_e, ["positive electrode"]),
        )
        self.variables = self.get_variables(c_e_var, N_e)

        # Cut off if concentration goes negative
        self.events = [pybamm.Function(np.min, c_e)]

    def get_variables(self, c_e, N_e):
        """
        Calculate dimensionless and dimensional variables for the electrolyte diffusion
        submodel

        Parameters
        ----------
        c_e : :class:`pybamm.Concatenation`
            Electrolyte concentration
        N_e : :class:`pybamm.Symbol`
            Flux of electrolyte cations

        Returns
        -------
        dict
            Dictionary {string: :class:`pybamm.Symbol`} of relevant variables
        """
        c_e_typ = self.set_of_parameters.c_e_typ

        if c_e.domain == []:
            c_e_n = pybamm.Broadcast(c_e, domain=["negative electrode"])
            c_e_s = pybamm.Broadcast(c_e, domain=["separator"])
            c_e_p = pybamm.Broadcast(c_e, domain=["positive electrode"])
            c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        c_e_n, c_e_s, c_e_p = c_e.orphans
        return {
            "Electrolyte concentration": c_e,
            "Negative electrolyte concentration": c_e_n,
            "Separator electrolyte concentration": c_e_s,
            "Positive electrolyte concentration": c_e_p,
            "Reduced cation flux": N_e,
            "Electrolyte concentration [mols m-3]": c_e_typ * c_e,
            "Negative electrolyte concentration [mols m-3]": c_e_typ * c_e_n,
            "Separator electrolyte concentration [mols m-3]": c_e_typ * c_e_s,
            "Positive electrolyte concentration [mols m-3]": c_e_typ * c_e_p,
        }
