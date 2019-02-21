#
# Single Particle Model with Electrolyte (SPMe)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class SPMe(pybamm.BaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.

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

        "Model Variables"
        # Electrolyte concentration
        # c_en = pybamm.Variable("c_en", ["negative electrode"])
        # c_es = pybamm.Variable("c_es", ["separator"])
        # c_ep = pybamm.Variable("c_ep", ["positive electrode"])
        # c_e = pybamm.Concatenation(c_en, c_es, c_ep)
        c_e = pybamm.Variable(
            "c_e", ["negative electrode", "separator", "positive electrode"]
        )

        # Particle concentration
        c_n = pybamm.Variable("c_n", ["negative particle"])
        c_p = pybamm.Variable("c_p", ["positive particle"])

        "Model Parameters and functions"
        #

        "Interface Conditions"
        G_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        G_p = pybamm.interface.homogeneous_reaction(["positive electrode"])
        G = pybamm.Concatenation(G_n, pybamm.Scalar(0, domain=["separator"]), G_p)

        "Model Equations"

        self.update(
            pybamm.electrolyte_diffusion.StefanMaxwell(c_e, G),
            pybamm.particle.Standard(c_n, G_n),
            pybamm.particle.Standard(c_p, G_p),
        )

        "Additional Conditions"
        # phi is only determined to a constant so set phi_n = 0 on left boundary
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        # TODO: add voltage and overpotentials to this
        additional_variables = {}
        self._variables.update(additional_variables)
