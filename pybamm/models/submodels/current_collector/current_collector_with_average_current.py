#
# Class for an current collector model where the sink current is equal to the average
# current
#

import pybamm
from ..current_collector import BaseModel


class AverageCurrent(BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the current collector resistance and potentials.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def get_fundamental_variables(self):

        R_cn = pybamm.standard_variables.R_cn
        R_cp = pybamm.standard_variables.R_cp

        param = pybamm.standard_parameters_lithium_ion
        l_cn = param.l_cn
        l_cp = param.l_cp
        l_y = param.l_y
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        delta = param.delta

        I_app = pybamm.electrical_parameters.current_with_time * l_y

        phi_s_cn = R_cn * (I_app / l_y) / (l_cn * sigma_cn_dbl_prime)

        # phi_s_cp_red = phi_s_cp - V (we don't know V yet!)
        phi_s_cp_red = -R_cp * (I_app / l_y) / (l_cp * sigma_cp_dbl_prime)

        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        phi_s_cn_av = pybamm.Integral(phi_s_cn, [y, z])
        phi_s_cp_red_av = pybamm.Integral(phi_s_cp_red, [y, z])

        # average current collector resistance:
        R_cc = -(phi_s_cn_av + phi_s_cp_red_av) / I_app

        # average current collector Ohmic losses
        Delta_Phi_cc = -delta * I_app * R_cc

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        ptl_scale = param.potential_scale
        I_typ = param.I_typ

        variables.update(
            {
                "Negative current collector resistance": R_cn,
                "Positive current collector resistance": R_cp,
                "Reduced positive current collector potential": phi_s_cp_red,
                "Reduced positive current collector potential [V]": phi_s_cp_red
                * ptl_scale,
                "Average current collector ohmic losses": Delta_Phi_cc,
                "Effective current collector resistance": R_cc,
                "Effective current collector resistance [Ohm]": R_cc
                * ptl_scale
                / I_typ,
            }
        )

        return variables

    def get_coupled_variables(self, variables):

        phi_s_n_av = variables["Negative electrode potential"]
        phi_s_p_av = variables["Positive electrode potential"]
        Delta_Phi_cc = variables["Average current collector ohmic losses"]
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp_red = variables["Reduced positive current collector potential"]

        V_av = pybamm.boundary_value(phi_s_p_av, "right") - pybamm.boundary_value(
            phi_s_n_av, "left"
        )

        V = V_av + Delta_Phi_cc
        phi_s_cp = phi_s_cp_red + V

        variables.update(self._get_standard_potential_variables(phi_s_cn, phi_s_cp))

    def set_algebraic(self, variables):
        R_cn = variables["Negative current collector resistance"]
        R_cp = variables["Positive current collector resistance"]

        self.algebraic = {
            R_cn: pybamm.Laplacian(R_cn) - 1,
            R_cp: pybamm.Laplacian(R_cp) - 1,
        }

    def set_boundary_conditions(self, variables):
        R_cn = variables["Negative current collector resistance"]
        R_cp = variables["Positive current collector resistance"]

        self.boundary_conditions = {
            R_cn: {
                "negative tab": (pybamm.Scalar(0), "Dirichlet"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            },
            R_cp: {
                "negative tab": (pybamm.Scalar(0), "Neumann"),
                "positive tab": (pybamm.Scalar(0), "Dirichlet"),
            },
        }

    def set_initial_conditions(self, variables):
        # initial guess for solver
        R_cn = variables["Negative current collector resistance"]
        R_cp = variables["Positive current collector resistance"]

        self.initial_conditions = {
            R_cn: pybamm.Scalar(0),
            R_cp: pybamm.Scalar(0),
        }

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    @property
    def default_geometry(self):
        return pybamm.Geometry("2D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 32, var.z: 32}

    @property
    def default_submesh_types(self):
        return {
            "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh)
        }

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.ScikitFiniteElement()}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()
