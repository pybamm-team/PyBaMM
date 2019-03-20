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
    epsilon : :class:`pybamm.Symbol`
        The (electrolyte/liquid phase) porosity (can be Variable or Parameter)
    j : :class:`pybamm.Symbol`
        An expression tree that represents the interfacial current density at the
        electrode-electrolyte interface
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`BaseModel`
    """

    def __init__(self, phi_s, eps, j, param):
        super().__init__()

        current = pybamm.standard_parameters.current_with_time

        # algebraic model only
        self.rhs = {}

        # different bounday conditions in each electrode
        if phi_s.domain == ["negative electrode"]:
            i_n = -param.sigma_n * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.grad(i_n) + j}
            self.boundary_conditions = {i_n: {"left": current, "right": 0}}
            self.initial_conditions = {phi_s: 0}
            self.variables = {
                "Negative electrode solid potential": phi_s,
                "Negative electrode solid current": i_n,
            }
        elif phi_s.domain == ["positive electrode"]:
            i_p = -param.sigma_p * (1 - eps) ** param.b * pybamm.grad(phi_s)
            self.algebraic = {phi_s: pybamm.grad(i_p) + j}
            self.boundary_conditions = {i_p: {"left": 0, "right": current}}
            self.initial_conditions = {phi_s: 0}
            self.variables = {
                "Positive electrode solid potential": phi_s,
                "Positive electrode solid current": i_p,
            }
        # for whole cell domain call both electrode models and ignore separator
        elif phi_s.domain == ["negative electrode", "separator", "positive electrode"]:
            phi_s_n, phi_s_s, phi_s_p = phi_s.orphans
            eps_n, eps_s, eps_p = eps.orphans
            j_n, j_s, j_p = j.orphans
            neg_model = Ohm(phi_s_n, eps_n, j_n, param)
            pos_model = Ohm(phi_s_p, eps_p, j_p, param)
            self.update(neg_model, pos_model)
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(phi_s.domain))
