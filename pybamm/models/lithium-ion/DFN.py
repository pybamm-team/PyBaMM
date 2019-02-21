#
# Lead-acid LOQS model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class DFN(pybamm.BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.

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

        whole_cell = ["negative electrode", "separator", "positive electrode"]

        "Model Variables"
        # Electrolyte concentration
        c_en = pybamm.Variable("c_en", ["negative electrode"])
        c_es = pybamm.Variable("c_es", ["separator"])
        c_ep = pybamm.Variable("c_ep", ["positive electrode"])
        # TODO: find the right way to concatenate
        c_e = pybamm.Concatenation(c_en, c_es, c_ep)

        # Electrolyte Potential
        phi_en = pybamm.Variable("phi_en", ["negative electrode"])
        phi_es = pybamm.Variable("phi_es", ["separator"])
        phi_ep = pybamm.Variable("phi_ep", ["positive electrode"])
        # TODO: find the right way to concatenate
        phi_e = pybamm.Concatenation(phi_en, phi_es, phi_ep)

        # Electrode Potential
        phi_n = pybamm.Variable("phi_n", ["negative electrode"])
        phi_p = pybamm.Variable("phi_p", ["positive electrode"])

        # Particle concentration
        c_en = pybamm.Variable("c_en", ["negative electrode"])
        c_ep = pybamm.Variable("c_ep", ["positive electrode"])

        "Model Parameters"

        "Model Equations"

