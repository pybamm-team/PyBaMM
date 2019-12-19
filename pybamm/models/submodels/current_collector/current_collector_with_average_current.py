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

        phi_s_cn = delta * R_cn * (I_app / l_y) / (l_cn * sigma_cn_dbl_prime)

        # phi_s_cp_red = phi_s_cp - V (we don't know V yet!)
        phi_s_cp_red = -delta * R_cp * I_app / (l_y * l_cp * sigma_cp_dbl_prime)

        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        phi_s_cn_av = pybamm.Integral(phi_s_cn, [y, z]) / l_y
        phi_s_cp_red_av = pybamm.Integral(phi_s_cp_red, [y, z]) / l_y

        # average current collector resistance:
        R_cc = -(phi_s_cn_av - phi_s_cp_red_av) / I_app

        # average current collector Ohmic losses
        Delta_Phi_cc = -I_app * R_cc  # delta already added to phi

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # add in current collector current densities
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.PrimaryBroadcast(
            self.param.current_with_time, "current collector"
        )
        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

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
                "Average current collector ohmic losses [V]": Delta_Phi_cc * ptl_scale,
                "Effective current collector resistance": R_cc,
                "Effective current collector resistance [Ohm]": R_cc
                * ptl_scale
                / I_typ,
            }
        )

        # Hack to get the leading-order current collector current density
        # Note that this should be different from the actual (composite) current
        # collector current density for 2+1D models, but not sure how to implement this
        # using current structure of lithium-ion models
        variables["Leading-order current collector current density"] = variables[
            "Current collector current density"
        ]

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

        return variables

    def set_algebraic(self, variables):
        R_cn = variables["Negative current collector resistance"]
        R_cp = variables["Positive current collector resistance"]

        self.algebraic = {
            R_cn: pybamm.laplacian(R_cn) - pybamm.source(1, R_cn),
            R_cp: pybamm.laplacian(R_cp) - pybamm.source(1, R_cp),
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

        # add on boundary conditions for phi as well for use in the
        # temperature equations. Should be able to remove the need
        # to do this at some stage
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        V = pybamm.BoundaryValue(phi_s_cp, "positive tab")

        param = pybamm.standard_parameters_lithium_ion
        applied_current = param.current_with_time
        cc_area = self._get_effective_current_collector_area()

        pos_tab_bc = (
            -applied_current
            * cc_area
            / (param.sigma_cp * param.delta ** 2 * param.l_cp)
        )

        # Boundary condition needs to be on the variables that go into the Laplacian,
        # even though phi_s_cp isn't a pybamm.Variable object
        self.boundary_conditions.update(
            {
                phi_s_cn: {
                    "negative tab": (pybamm.Scalar(0), "Dirichlet"),
                    "positive tab": (pybamm.Scalar(0), "Neumann"),
                },
                phi_s_cp: {
                    "negative tab": (pybamm.Scalar(0), "Neumann"),
                    "positive tab": (pos_tab_bc, "Neumann"),
                },
            }
        )

    def set_initial_conditions(self, variables):
        # initial guess for solver
        R_cn = variables["Negative current collector resistance"]
        R_cp = variables["Positive current collector resistance"]

        self.initial_conditions = {
            R_cn: pybamm.Scalar(0),
            R_cp: pybamm.Scalar(0),
        }

    def _get_effective_current_collector_area(self):
        "Return the area of the current collector"
        return self.param.l_y * self.param.l_z

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
