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

    def __init__(self, phi, eps, j, param):
        super().__init__()

        self.rhs = {}
        self.initial_conditions = {}
        if phi.domain == ["negative electrode"]:
            i = -param.sigma_n * (1 - eps) ** param.b * pybamm.grad(phi)
            self.boundary_conditions = {i: {"left": i_cell, "right": 0}}
            self.variables = {
                "Negative electrode solid potential": phi,
                "Negative electrode solid current": i,
            }
        elif phi.domain == ["positive electrode"]:
            i = -param.sigma_p * (1 - eps) ** param.b * pybamm.grad(phi)
            self.boundary_conditions = {i: {"left": 0, "right": i_cell}}
            self.variables = {
                "Positive electrode solid potential": phi,
                "Positive electrode solid current": i,
            }
        else:
            raise pybamm.DomainError("domain '{}' not recognised".format(domain))
        self.algebraic = {phi: pybamm.grad(i) + j}
