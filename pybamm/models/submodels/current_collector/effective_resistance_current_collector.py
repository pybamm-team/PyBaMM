#
# Class for calcuting the effective resistance of two-dimensional current collectors
#
import pybamm


class SingleParticlePotentialPair(pybamm.BaseModel):
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
        psi = pybamm.Variable(
            "Current collector potential weighted sum", ["current collector"]
        )
        W = pybamm.Variable(
            "Perturbation to current collector potential difference",
            ["current collector"],
        )
        c_psi = pybamm.Variable(
            "Lagrange multiplier for variable `psi`"
        )
        c_W = pybamm.Variable(
            "Lagrange multiplier for variable `W`"
        )

        self.variables = {
            "Current collector potential weighted sum": psi,
            "Perturbation to current collector potential difference": W,
            "Lagrange multiplier for variable `psi`": c_psi,
            "Lagrange multiplier for variable `W`": c_W,
        }

        # Algebraic equations (enforce zero mean constraint through Lagrange multiplier)
        self.algebraic = {
            psi: pybamm.laplacian(psi)
            + c_psi * pybamm.DefiniteIntegralVector(psi, vector_type="column"),
            W: pybamm.laplacian(W)
            - pybamm.source(1, W)
            + c_W * pybamm.DefiniteIntegralVector(W, vector_type="column"),
            c_psi: pybamm.DefiniteIntegralVector(psi, vector_type="row"),
            c_W: pybamm.DefiniteIntegralVector(W, vector_type="row"),
        }

        # Boundary conditons
        self.boundary_conditions = {
            psi: {
                "left": (pybamm.Scalar(param.l_cn), "Neumann"),
                "right": (pybamm.Scalar(-param.l_cp), "Neumann"),
            },
            W: {
                "left": (pybamm.Scalar(), "Neumann"),
                "right": (pybamm.Scalar(), "Neumann"),
            },
        }
