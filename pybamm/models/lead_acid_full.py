#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

# import pybamm


# class LeadAcid(pybamm.BaseModel):
#     """One-dimensional model for lead-acid.
#
#     Attributes
#     ----------
#
#     rhs: dict
#         A dictionary that maps expressions (variables) to expressions that represent
#         the rhs
#     initial_conditions: dict
#         A dictionary that maps expressions (variables) to expressions that represent
#         the initial conditions
#     boundary_conditions: dict
#         A dictionary that maps expressions (variables) to expressions that represent
#         the boundary conditions
#     """
#
#     def __init__(self):
#         # Variables
#         c = pybamm.Variable("c", domain=["whole cell"])
#         epsn = pybamm.Variable("epsn", domain=["negative electrode"])
#         epss = pybamm.Variable("epss", domain=["separator"])
#         epsp = pybamm.Variable("epsp", domain=["positive electrode"])
#         eps = pybamm.Concatenation(epsn, epss, epsp)
#         phin = pybamm.Variable("phin", domain=["negative electrode"])
#         phip = pybamm.Variable("phip", domain=["positive electrode"])
#         j = pybamm.interface.ButlerVolmer()
#
#         # Parameters
#         ln = pybamm.standard_parameters.ln
#         ls = pybamm.standard_parameters.ls
#         lp = pybamm.standard_parameters.lp
#         s = pybamm.standard_parameters.s
#         beta_surf = pybamm.standard_parameters.beta_surf
#         # Functions
#         D = pybamm.standard_parameters.D(c)
#         chi = pybamm.standard_parameters.chi(c)
#         kappa = pybamm.standard_parameters.kappa(c)
#
#         # Initial conditions
#         cinit = pybamm.standard_parameters.cinit
#         epsinit = pybamm.standard_parameters.epsinit
#
#         # ODEs
#         # porosity
#         depsdt = -beta_surf * j
#         # concentration
#         N = -D * pybamm.grad(c)
#         dcdt = 1 / eps * (-1 / Cd * pybamm.div(N) + s * j - c * depsdt)
#         # current
#         i_n = 0
#         i_p = 0
#         dphindt = 1 / gamma_dl_n * (pybamm.div(i_n) - An * jn)
#         dphipdt = 1 / gamma_dl_p * (pybamm.div(i_p) - Ap * jp)
#         self.rhs = {c: dcdt, eps: depsdt, phin: dphindt, phip: dphipdt}
#         # Initial conditions
#         self.initial_conditions = {c: cinit, eps: epsinit}
#         # ODE model -> no boundary conditions
#         self.boundary_conditions = {
#             N: {"left": 0, "right": 0},
#             i_n: {"left": 0, "right": icell},
#             i_p: {"left": icell, "right": 0},
#         }
#
#         # Variables
#         self.variables = {
#             "concentration": c,
#             "porosity": eps,
#             "interfacial current density": j,
#             "electrolyte current density": i,
#             "solid current density": icell - i,
#             "solid potential": Phis,
#             "voltage": V,
#         }
