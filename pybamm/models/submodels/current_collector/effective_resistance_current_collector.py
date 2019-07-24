#
# Class for calcuting the effective resistance of two-dimensional current collectors
#
import pybamm
import os


class EfectiveResistance2D(pybamm.BaseModel):
    """A model which calculates the effective Ohmic resistance of the current
    collectors in the limit of large electrical conductivity.
    Note:  This submodel should be solved before a one-dimensional model to calculate
    and return the parameter "Effective current collector Ohmic resistance"

    Parameters
    ----------
    """

    def __init__(self, param):
        super().__init__(param)
        self.name = "Effective resistance in current collector model"
        self.param = param

        # Set variables
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

        # TODO: get potentials

        # Define effective current collector resistance
        l_cn = param.l_cn
        l_cp = param.l_cp
        l_y = param.l_y
        delta = param.delta
        sigma_cn_dbl_prime = param.sigma_cn * delta ** 2
        sigma_cp_dbl_prime = param.sigma_cp * delta ** 2
        alpha_prime = 1 / (sigma_cn_dbl_prime * delta * l_cn) + 1 / (
            sigma_cp_dbl_prime * delta * l_cp
        )

        R_cc = (
            (alpha_prime / l_y)
            * (
                sigma_cn_dbl_prime * l_cn * pybamm.BoundaryValue(W, "right")
                + sigma_cp_dbl_prime * l_cp * pybamm.BoundaryValue(W, "left")
            )
            - (pybamm.BoundaryValue(psi, "right") - pybamm.BoundaryValue(psi, "left"))
        ) / (sigma_cn_dbl_prime * l_cn + sigma_cp_dbl_prime * l_cp)

        self.variables = {
            "Current collector potential weighted sum": psi,
            "Perturbation to current collector potential difference": W,
            "Lagrange multiplier for variable `psi`": c_psi,
            "Lagrange multiplier for variable `W`": c_W,
            "Effective current collector resistance": R_cc,
        }

        # Algebraic equations (enforce zero mean constraint through Lagrange multiplier)
        self.algebraic = {
            psi: pybamm.laplacian(psi)
            + c_psi * pybamm.DefiniteIntegralVector(psi, vector_type="column"),
            W: pybamm.laplacian(W)
            - pybamm.source(1, W)
            + c_W * pybamm.DefiniteIntegralVector(W, vector_type="column"),
            c_psi: pybamm.Integral(psi, [var.y, var.z]),
            c_W: pybamm.Integral(W, [var.y, var.z]),
        }

        # Boundary conditons
        W_neg_tab_bc = pybamm.Scalar(l_y / (alpha_prime * sigma_cn_dbl_prime))
        W_pos_tab_bc = pybamm.Scalar(l_y / (alpha_prime * sigma_cp_dbl_prime))

        self.boundary_conditions = {
            psi: {
                "left": (pybamm.Scalar(l_cn), "Neumann"),
                "right": (pybamm.Scalar(-l_cp), "Neumann"),
            },
            W: {"left": (W_neg_tab_bc, "Neumann"), "right": (W_pos_tab_bc, "Neumann")},
        }

    @property
    def default_parameter_values(self):
        # Defualt li-ion parameter values
        input_path = os.path.join(
            pybamm.root_dir(), "input", "parameters", "lithium-ion"
        )
        return pybamm.ParameterValues(
            os.path.join(
                input_path, "mcmb2528_lif6-in-ecdmc_lico2_parameters_Dualfoil.csv"
            )
        )

    @property
    def default_geometry(self):
        return pybamm.Geometry("2D current collector")

    @property
    def default_var_pts(self):
        var = pybamm.standard_spatial_vars
        return {var.y: 10, var.z: 10}

    @property
    def default_submesh_types(self):
        return {"current collector": pybamm.Scikit2DSubMesh}

    @property
    def default_spatial_methods(self):
        return {"current collector": pybamm.ScikitFiniteElement}

    @property
    def default_solver(self):
        return pybamm.AlgebraicSolver()
