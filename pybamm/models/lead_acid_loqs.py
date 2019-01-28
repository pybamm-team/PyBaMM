#
# Reaction-diffusion model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LeadAcidLOQS(pybamm.BaseModel):
    """Leading-Order Quasi-Static model for lead-acid.

    Attributes
    ----------

    rhs: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the rhs
    initial_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the initial conditions
    boundary_conditions: dict
        A dictionary that maps expressions (variables) to expressions that represent
        the boundary conditions
    """

    def __init__(self):
        # Variables
        c = pybamm.Variable("c", domain=["time only"])
        epsn = pybamm.Variable("epsn", domain=["time only"])
        epss = pybamm.Variable("epss", domain=["time only"])
        epsp = pybamm.Variable("epsp", domain=["time only"])
        jn = pybamm.interface.HomogeneousReactions()
        jp = pybamm.interface.HomogeneousReactions()

        # Parameters
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls
        lp = pybamm.standard_parameters.lp
        sn = pybamm.standard_parameters.sn
        sp = pybamm.standard_parameters.sp
        beta_surf_n = pybamm.standard_parameters.beta_surf_n
        beta_surf_p = pybamm.standard_parameters.beta_surf_p
        # Initial conditions
        cinit = pybamm.standard_parameters.cinit
        epsninit = pybamm.standard_parameters.epsninit
        epssinit = pybamm.standard_parameters.epssinit
        epspinit = pybamm.standard_parameters.epspinit

        # ODEs
        depsndt = -beta_surf_n * j
        depssdt = -beta_surf_p * j
        dcdt = (
            1
            / (ln * epsn + ls * epss + lp * epsp)
            * ((sn - sp) - c * (ln * depsndt + lp * depspdt))
        )
        self.rhs = {c: dcdt, epsn: depsndt, epss: pybamm.Scalar(0), epsp: depspdt}
        # Initial conditions
        self.initial_conditions = {
            c: cinit,
            epsn: epsninit,
            epss: epssinit,
            epsp: epspinit,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
