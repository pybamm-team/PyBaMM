#
# Doyle-Fuller-Newman (DFN) Model
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
        c_n = pybamm.Variable("c_n", ["negative particle"])
        c_p = pybamm.Variable("c_p", ["positive particle"])

        "Model Parameters and functions"
        m_n = pybamm.standard_parameters.m_n
        m_p = pybamm.standard_parameters.m_p
        U_n = pybamm.standard_parameters.U_n
        U_p = pybamm.standard_parameters.U_p

        "Interface Conditions"
        cn_surf = pybamm.SurfaceValue(c_n)
        cp_surf = pybamm.SurfaceValue(c_p)
        G_n = pybamm.interface.butler_volmer(
            m_n, U_n, c_en, phi_n - phi_en, ck_surf=cn_surf
        )
        G_p = pybamm.interface.butler_volmer(
            m_p, U_p, c_ep, phi_p - phi_ep, ck_surf=cp_surf
        )
        G = pybamm.Concatenation(G_n, pybamm.Scalar(0, domain=["separator"]), G_p)

        "Model Equations"
        self.update(
            pybamm.electrolyte_diffusion.StefanMaxwell(c_e, G),
            pybamm.electrolyte_current.StefanMaxwell(c_e, phi_e, G),
            pybamm.electrode.Standard(phi_n, G_n),
            pybamm.electrode.Standard(phi_p, G_p),
            pybamm.particle.Standard(c_n, G_n),
            pybamm.particle.Standard(c_p, G_p),
        )

        "Additional Conditions"
        # phi is only determined to a constant so set phi_n = 0 on left boundary
        additional_bcs = {phi_n: {"left": pybamm.Scalar(0)}}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        # TODO: add voltage and overpotentials to this
        additional_variables = {}
        self._variables.update(additional_variables)
