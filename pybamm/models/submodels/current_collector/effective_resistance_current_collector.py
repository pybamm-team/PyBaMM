#
# Class for calcuting the effective resistance of two-dimensional current collectors
#
import pybamm
import os


class EffectiveResistance2D(pybamm.BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the effective current collector resistance.

    """

    def __init__(self):
        super().__init__()
        self.name = "Effective resistance in current collector model"
        self.param = pybamm.standard_parameters_lithium_ion

        # Get useful parameters
        param = self.param
        l_cn = param.l_cn
        l_cp = param.l_cp
        l_y = param.l_y
        sigma_cn_dbl_prime = param.sigma_cn_dbl_prime
        sigma_cp_dbl_prime = param.sigma_cp_dbl_prime
        alpha_prime = param.alpha_prime

        # Set model variables
        var = pybamm.standard_spatial_vars

        psi = pybamm.Variable(
            "Current collector potential weighted sum", ["current collector"]
        )
        W = pybamm.Variable(
            "Perturbation to current collector potential difference",
            ["current collector"],
        )
        c_psi = pybamm.Variable("Lagrange multiplier for variable `psi`")
        c_W = pybamm.Variable("Lagrange multiplier for variable `W`")

        self.variables = {
            "Current collector potential weighted sum": psi,
            "Perturbation to current collector potential difference": W,
            "Lagrange multiplier for variable `psi`": c_psi,
            "Lagrange multiplier for variable `W`": c_W,
        }

        # Algebraic equations (enforce zero mean constraint through Lagrange multiplier)
        # 0*LagrangeMultiplier hack otherwise gives KeyError
        self.algebraic = {
            psi: pybamm.laplacian(psi)
            + c_psi * pybamm.DefiniteIntegralVector(psi, vector_type="column"),
            W: pybamm.laplacian(W)
            - pybamm.source(1, W)
            + c_W * pybamm.DefiniteIntegralVector(W, vector_type="column"),
            c_psi: pybamm.Integral(psi, [var.y, var.z]) + 0 * c_psi,
            c_W: pybamm.Integral(W, [var.y, var.z]) + 0 * c_W,
        }

        # Boundary conditons
        psi_neg_tab_bc = l_cn
        psi_pos_tab_bc = -l_cp
        W_neg_tab_bc = l_y / (alpha_prime * sigma_cn_dbl_prime)
        W_pos_tab_bc = l_y / (alpha_prime * sigma_cp_dbl_prime)

        self.boundary_conditions = {
            psi: {
                "left": (psi_neg_tab_bc, "Neumann"),
                "right": (psi_pos_tab_bc, "Neumann"),
            },
            W: {"left": (W_neg_tab_bc, "Neumann"), "right": (W_pos_tab_bc, "Neumann")},
        }

        # "Initial conditions" provides initial guess for solver
        # TODO: better guess than zero?
        self.initial_conditions = {
            psi: pybamm.Scalar(0),
            W: pybamm.Scalar(0),
            c_psi: pybamm.Scalar(0),
            c_W: pybamm.Scalar(0),
        }

        # Define effective current collector resistance
        R_cc = (
            (alpha_prime / l_y)
            * (
                sigma_cn_dbl_prime * l_cn * pybamm.BoundaryValue(W, "right")
                + sigma_cp_dbl_prime * l_cp * pybamm.BoundaryValue(W, "left")
            )
            - (pybamm.BoundaryValue(psi, "right") - pybamm.BoundaryValue(psi, "left"))
        ) / (sigma_cn_dbl_prime * l_cn + sigma_cp_dbl_prime * l_cp)

        R_cc_dim = R_cc * param.potential_scale / param.I_typ

        self.variables.update(
            {
                "Effective current collector resistance": R_cc,
                "Effective current collector resistance [Ohm]": R_cc_dim,
            }
        )

    def get_processed_potentials(self, solution, mesh, param_values, V_av, I_av):
        """
        Calculates the potentials in the current collector given
        the average voltage and current.
        Note: This takes in the *processed* V_av and I_av from a 1D simulation
        representing the average cell behaviour and returns a dictionary of
        processed potentials.
        """
        # Get required processed parameters
        param = self.param
        l_cn = param_values.process_symbol(param.l_cn).evaluate()
        l_cp = param_values.process_symbol(param.l_cp).evaluate()
        l_y = param_values.process_symbol(param.l_y).evaluate()
        l_z = param_values.process_symbol(param.l_z).evaluate()
        neg_tab_y = param_values.process_symbol(param.centre_y_tab_n).evaluate()
        neg_tab_z = param_values.process_symbol(param.centre_z_tab_n).evaluate()
        sigma_cn_prime = param_values.process_symbol(param.sigma_cn_prime).evaluate()
        sigma_cp_prime = param_values.process_symbol(param.sigma_cp_prime).evaluate()
        alpha = param_values.process_symbol(param.alpha).evaluate()
        pot_scale = param_values.process_symbol(param.potential_scale).evaluate()
        U_ref = param_values.process_symbol(param.U_p_ref - param.U_n_ref).evaluate()

        # Process psi and W
        psi = pybamm.ProcessedVariable(
            self.variables["Current collector potential weighted sum"],
            solution.t,
            solution.y,
            mesh,
        )
        W = pybamm.ProcessedVariable(
            self.variables["Perturbation to current collector potential difference"],
            solution.t,
            solution.y,
            mesh,
        )

        # Create callable combination of ProcessedVariable objects for potentials
        def V_cc(t, y, z):
            return V_av(t) - alpha * I_av(t) * W(t=None, y=y, z=z)

        def V_cc_dim(t, y, z):
            return U_ref + V_cc(t, y, z) * pot_scale

        denominator = sigma_cn_prime * l_cn + sigma_cn_prime * l_cp

        # The method onlu defines psi up to an arbitrray function of time. This
        # is fixed by ensuring phi_s_cn = 0 on the negative tab when reconstructing
        # the potentials
        def phi_s_cn_tab(t):
            phi_s_cn_tab = (
                I_av(t) * l_y * l_z * psi(t=None, y=neg_tab_y, z=neg_tab_z)
                - sigma_cp_prime * l_cp * V_cc(t, y=neg_tab_y, z=neg_tab_z)
            ) / denominator
            return phi_s_cn_tab

        def phi_s_cn(t, y, z):
            phi_s_cn = (
                I_av(t) * l_y * l_z * psi(t=None, y=y, z=z)
                - sigma_cp_prime * l_cp * V_cc(t, y, z)
            ) / denominator
            return phi_s_cn - phi_s_cn_tab(t)

        def phi_s_cn_dim(t, y, z):
            return phi_s_cn(t, y, z) * pot_scale

        def phi_s_cp(t, y, z):
            phi_s_cp = (
                I_av(t) * l_y * l_z * psi(t=None, y=y, z=z)
                + sigma_cn_prime * l_cn * V_cc(t, y, z)
            ) / denominator
            return phi_s_cp - phi_s_cn_tab(t)

        def phi_s_cp_dim(t, y, z):
            return U_ref + phi_s_cp(t, y, z) * pot_scale

        potentials = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn_dim,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": phi_s_cp_dim,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": V_cc_dim,
        }
        return potentials

    @property
    def default_parameter_values(self):
        # Defualt li-ion parameter values
        input_path = os.path.join(
            pybamm.root_dir(), "input", "parameters", "lithium-ion"
        )
        return pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            ),
            {
                "Typical current [A]": 1,
                "Current function": pybamm.GetConstantCurrent(
                    pybamm.standard_parameters_lithium_ion.I_typ
                ),
            },
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("2D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 32, var.z: 32}

    @property
    def default_submesh_types(self):
        return {"current collector": pybamm.Scikit2DSubMesh}

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.ScikitFiniteElement}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()
