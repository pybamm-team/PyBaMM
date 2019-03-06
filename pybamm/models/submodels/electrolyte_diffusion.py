#
# Equation classes for the electrolyte
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class StefanMaxwell(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface

    *Extends:* :class:`BaseModel`
    """

<<<<<<< HEAD:pybamm/models/submodels/electrolyte.py
    def __init__(self, j):
        super().__init__()

        # Parameters
        sp = pybamm.standard_parameters
        spli = pybamm.standard_parameters_lithium_ion

        electrolyte_domain = ["negative electrode", "separator", "positive electrode"]

        c_e = pybamm.Variable("c_e", electrolyte_domain)

        N_e = -sp.D_e(c_e) * (spli.epsilon ** sp.b) * pybamm.grad(c_e)
=======
    def __init__(self, c_e, G):
        super().__init__()

        # TODO: spatially dependent
        epsilon = pybamm.standard_parameters.epsilon_s
        b = pybamm.standard_parameters.b
        delta = pybamm.standard_parameters.delta
        nu = pybamm.standard_parameters.nu
        t_plus = pybamm.standard_parameters.t_plus
        ce0 = pybamm.standard_parameters.ce0

        N_e = -(epsilon ** b) * pybamm.grad(c_e)
>>>>>>> issue-74-implement-DFN:pybamm/models/submodels/electrolyte_diffusion.py

        self.rhs = {
            c_e: -pybamm.div(N_e) / spli.C_e / spli.epsilon
            + sp.s / spli.gamma_hat_e * j
        }
        self.initial_conditions = {c_e: spli.c_e_init}
        self.boundary_conditions = {
            N_e: {"left": pybamm.Scalar(0), "right": pybamm.Scalar(0)}
        }
        self.variables = {"c_e": c_e, "N_e": N_e}


class StefanMaxwellDiffusionWithPorosity(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, c_e, epsilon, j, param):
        super().__init__()
        sp = pybamm.standard_parameters

        # Flux
        N_e = -(epsilon ** sp.b) * pybamm.grad(c_e)
        # Porosity change
        deps_dt = -param.beta_surf * j

        # Model
        self.rhs = {
            c_e: 1
            / epsilon
            * (
                -pybamm.div(N_e) / param.C_e
                + sp.s / param.gamma_hat_e * j
                - c_e * deps_dt
            )
        }
        self.initial_conditions = {c_e: param.c_e_init}
        self.boundary_conditions = {N_e: {"left": 0, "right": 0}}
        self.variables = {"c_e": c_e, "N_e": N_e}


class Porosity(pybamm.BaseModel):
    """A class that generates the expression tree for Stefan-Maxwell Diffusion in the
    electrolyte.

    Parameters
    ----------
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, epsilon, j):
        super().__init__()
        sp = pybamm.standard_parameters_lead_acid

        # Model
        self.rhs = {epsilon: -sp.beta_surf * j}
        self.initial_conditions = {epsilon: sp.eps_init}
