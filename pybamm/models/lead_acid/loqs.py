#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class LOQS(pybamm.BaseModel):
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
        super().__init__()

        # Variables
        c0 = pybamm.Variable("c0", domain=[])
        eps0n = pybamm.Variable("eps0n", domain=[])
        eps0s = pybamm.Variable("eps0s", domain=[])
        eps0p = pybamm.Variable("eps0p", domain=[])

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
        U_Pb = pybamm.standard_parameters_lead_acid.U_Pb_ref
        U_PbO2 = pybamm.standard_parameters_lead_acid.U_PbO2_ref
        icell = pybamm.standard_parameters_lead_acid.icell
        # Initial conditions
        c0_init = pybamm.standard_parameters_lead_acid.c_init
        eps0n_init = pybamm.standard_parameters_lead_acid.epsn_init
        eps0s_init = pybamm.standard_parameters_lead_acid.epss_init
        eps0p_init = pybamm.standard_parameters_lead_acid.epsp_init

        # ODEs
        j0n = icell / ln
        j0p = -icell / lp
        deps0ndt = -beta_surf_n * j0n
        deps0pdt = -beta_surf_p * j0p
        dc0dt = (
            1
            / (ln * eps0n + ls * eps0s + lp * eps0p)
            * ((sn - sp) * icell - c0 * (ln * deps0ndt + lp * deps0pdt))
        )
        self.rhs = {
            c0: dc0dt,
            eps0n: deps0ndt,
            eps0s: pybamm.Scalar(0),
            eps0p: deps0pdt,
        }
        # Initial conditions
        self.initial_conditions = {
            c0: c0_init,
            eps0n: eps0n_init,
            eps0s: eps0s_init,
            eps0p: eps0p_init,
        }
        # ODE model -> no boundary conditions
        self.boundary_conditions = {}

        # Variables
        Phi0 = -U_Pb - j0n / (2 * iota_ref_n * c0)
        V0 = Phi0 + U_PbO2 - j0p / (2 * iota_ref_p * c0)
        # Phisn = pybamm.Scalar(0)
        # Phisp = V
        # Concatenate variables
        # eps = pybamm.Concatenation(epsn, epss, epsp)
        # Phis = pybamm.Concatenation(Phisn, pybamm.Scalar(0), Phisp)
        # self.variables = {"c": c, "eps": eps, "Phi": Phi, "Phis": Phis, "V": V}
        self.variables = {
            "c": pybamm.Broadcast(c0, ["whole cell"]),
            "Phi": pybamm.Broadcast(Phi0, ["whole cell"]),
            "V": V0,
            "int(epsilon_times_c)dx": (ln * eps0n + ls * eps0s + lp * eps0p) * c0,
        }

        # Overwrite default parameter values
        self.default_parameter_values = pybamm.ParameterValues(
            "input/parameters/lead-acid/default.csv", {"current scale": 1}
        )
