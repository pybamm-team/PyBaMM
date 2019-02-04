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
    variables: dict
        A dictionary that maps strings to expressions that represent
        the useful variables

    """

    def __init__(self):
        # Variables
        c = pybamm.Variable("c", domain=["time only"])
        epsn = pybamm.Variable("epsn", domain=["time only"])
        epss = pybamm.Variable("epss", domain=["time only"])
        epsp = pybamm.Variable("epsp", domain=["time only"])

        # Parameters
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls
        lp = pybamm.standard_parameters.lp
        sn = pybamm.standard_parameters_lead_acid.sn
        sp = pybamm.standard_parameters_lead_acid.sp
        beta_surf_n = pybamm.standard_parameters_lead_acid.beta_surf_n
        beta_surf_p = pybamm.standard_parameters_lead_acid.beta_surf_p
        iota_ref_n = pybamm.standard_parameters_lead_acid.iota_ref_n
        iota_ref_p = pybamm.standard_parameters_lead_acid.iota_ref_p
        U_Pb = pybamm.standard_parameters_lead_acid.U_Pb
        U_PbO2 = pybamm.standard_parameters_lead_acid.U_PbO2
        # Initial conditions
        c_init = pybamm.standard_parameters_lead_acid.c_init
        epsn_init = pybamm.standard_parameters_lead_acid.epsn_init
        epss_init = pybamm.standard_parameters_lead_acid.epss_init
        epsp_init = pybamm.standard_parameters_lead_acid.epsp_init

        # ODEs
        jn = pybamm.standard_parameters.icell / ln
        jp = -pybamm.standard_parameters.icell / lp
        depsndt = -beta_surf_n * jn
        depspdt = -beta_surf_p * jp
        dcdt = (
            1
            / (ln * epsn + ls * epss + lp * epsp)
            * ((sn - sp) - c * (ln * depsndt + lp * depspdt))
        )
        self.rhs = {c: dcdt, epsn: depsndt, epss: pybamm.Scalar(0), epsp: depspdt}
        # Initial conditions
        self.initial_conditions = {
            c: c_init,
            epsn: epsn_init,
            epss: epss_init,
            epsp: epsp_init,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        eps = pybamm.Concatenate(epsn, epss, epsp)
        Phisn = pybamm.Scalar(0)
        Phi = -U_Pb(c) - jn / (2 * iota_ref_n * c)
        V = -U_PbO2(c) - jp / (2 * iota_ref_p * c)
        Phi = pybamm.Concatenate(Phisn, pybamm.Scalar(np.nan), Phisp)
        self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
