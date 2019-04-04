#
# Equation classes for the electrode
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Ohm(pybamm.BaseModel):
    """Ohm's law + conservation of current for the current in the electrodes.

    Parameters
    ----------
    phi : :class:`pybamm.Symbol`
        The electric potential in the electrodes ("electrode potential")
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (optional)

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, phi_s, j, param, eps=None):
        super().__init__()

        current = pybamm.electrical_parameters.current_with_time

        # algebraic model only
        self.rhs = {}

        # different bounday conditions in each electrode
        if phi_s.domain == ["negative electrode"]:
            # if the porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon_n
            # liion sigma_n may already account for porosity
            i_s_n = -param.sigma_n * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_n) + j}
            self.boundary_conditions = {phi_s: {"left": 0}, i_s_n: {"right": 0}}
            self.initial_conditions = {phi_s: 0}
            self.variables = {
                "Negative electrode solid potential": phi_s,
                "Negative electrode solid current": i_s_n,
            }
        elif phi_s.domain == ["positive electrode"]:
            # if porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon_p
            # liion sigma_p may already account for porosity
            i_s_p = -param.sigma_p * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.div(i_s_p) + j}
            self.boundary_conditions = {i_s_p: {"left": 0, "right": current}}
            self.initial_conditions = {
                phi_s: param.U_p(param.c_p_init) - param.U_n(param.c_n_init)
            }
            self.variables = {
                "Positive electrode solid potential": phi_s,
                "Positive electrode solid current": i_s_p,
            }
        # for whole cell domain call both electrode models and ignore separator
        elif phi_s.domain == ["negative electrode", "separator", "positive electrode"]:
            # if porosity is not a variable, use the input parameter
            if eps is None:
                eps = param.epsilon
            phi_s_n, phi_s_s, phi_s_p = phi_s.orphans
            eps_n, eps_s, eps_p = eps.orphans
            j_n, j_s, j_p = j.orphans
            neg_model = Ohm(phi_s_n, j_n, param, eps=eps_n)
            pos_model = Ohm(phi_s_p, j_p, param, eps=eps_p)
            self.update(neg_model, pos_model)
            # Voltage variable
            voltage = pybamm.BoundaryValue(phi_s, "right") - pybamm.BoundaryValue(
                phi_s, "left"
            )
            self.variables.update({"Voltage": voltage})
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(phi_s.domain))

        # Set default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()


def explicit_solution_ohm(param, phi_e, ocp_p, eta_r_p, eps=None):
    # import standard spatial vairables
    x_n = pybamm.standard_spatial_vars.x_n
    x_p = pybamm.standard_spatial_vars.x_p

    # import geometric parameters
    l_n = pybamm.geometric_parameters.l_n
    l_p = pybamm.geometric_parameters.l_p

    # import current
    i_cell = param.current_with_time

    # if porosity is not passed in then use the parameter value
    if eps is None:
        eps = param.epsilon
    eps_n, eps_s, eps_p = [e.orphans[0] for e in eps.orphans]

    # extract right-most ocp, overpotential, and electrolyte potential
    ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
    eta_r_p_right = pybamm.BoundaryValue(eta_r_p, "right")
    phi_e_right = pybamm.BoundaryValue(phi_e, "right")

    # electrode potential
    phi_s_n = i_cell * x_n * (2 * l_n - x_n) / (2 * param.sigma_n * (1 - eps_n) * l_n)
    phi_s_s = pybamm.Broadcast(0, ["separator"])  # can we put NaN?

    phi_s_p = (
        ocp_p_right
        + eta_r_p_right
        + phi_e_right
        + i_cell
        * (1 - x_p)
        * (1 - 2 * l_p - x_p)
        / (2 * param.sigma_p * (1 - eps_p) * l_p)
    )
    phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

    # electrode current
    i_s_n = i_cell - i_cell * x_n / l_n
    i_s_s = pybamm.Broadcast(0, ["separator"])
    i_s_p = i_cell - i_cell * (1 - x_p) / l_p
    i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

    # average solid phase ohmic losses
    Delta_Phi_s = (
        -i_cell
        / 3
        * (l_p / param.sigma_p / (1 - eps_p) + l_n / param.sigma_n / (1 - eps_n))
    )

    return phi_s, i_s, Delta_Phi_s


def explicit_leading_order_ohm(param, phi_e, ocp_p, eta_r_p):

    # extract right-most ocp, overpotential, and electrolyte potential
    ocp_p_right = pybamm.BoundaryValue(ocp_p, "right")
    eta_r_p_right = pybamm.BoundaryValue(eta_r_p, "right")
    phi_e_right = pybamm.BoundaryValue(phi_e, "right")

    phi_s_n = pybamm.Scalar(0, ["negative electrode"])
    phi_s_s = pybamm.Scalar(0, ["separator"])
    phi_s_p = ocp_p_right + eta_r_p_right + phi_e_right
    phi_s_p.domain = ["positive electrode"]
    phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

    # electrode current
    i_s_n = i_cell - i_cell * x_n / l_n
    i_s_s = pybamm.Broadcast(0, ["separator"])
    i_s_p = i_cell - i_cell * (1 - x_p) / l_p
    i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

    Delta_Phi_s = pybamm.Scalar(0)

    return phi_s, i_s, Delta_Phi_s
