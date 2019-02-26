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
        whole_cell = ["negative electrode", "separator", "positive electrode"]

        # Electrolyte concentration

        # TODO: change once checkwellposedness changes
        # c_en = pybamm.Variable("c_en", ["negative electrode"])
        # c_es = pybamm.Variable("c_es", ["separator"])
        # c_ep = pybamm.Variable("c_ep", ["positive electrode"])
        # c_e = pybamm.Concatenation(c_en, c_es, c_ep)
        c_e = pybamm.Variable("c_e", whole_cell)

        # Particle concentration
        c_n = pybamm.Variable("c_n", ["negative particle"])
        c_p = pybamm.Variable("c_p", ["positive particle"])

        "Model Parameters and functions"
        # NOTE: is it better to just from standard_parameters import *?
        m_n = pybamm.standard_parameters.m_n
        m_p = pybamm.standard_parameters.m_p
        U_n = pybamm.standard_parameters.U_n
        U_p = pybamm.standard_parameters.U_p
        Lambda = pybamm.standard_parameters.Lambda
        C_hat_p = pybamm.standard_parameters.C_hat_p
        ln = pybamm.standard_parameters.ln
        ls = pybamm.standard_parameters.ls
        lp = pybamm.standard_parameters.lp
        delta = pybamm.standard_parameters.delta
        nu = pybamm.standard_parameters.nu
        epsilon_n = pybamm.standard_parameters.epsilon_n
        epsilon_s = pybamm.standard_parameters.epsilon_s
        epsilon_p = pybamm.standard_parameters.epsilon_p
        b = pybamm.standard_parameters.b
        sigma_e = pybamm.standard_parameters.sigma_e
        current = pybamm.standard_parameters.current
        t_plus = pybamm.standard_parameters.t_plus
        sigma_n = pybamm.standard_parameters.sigma_n
        sigma_p = pybamm.standard_parameters.sigma_p

        "Interface Conditions"
        G_n = pybamm.interface.homogeneous_reaction(["negative electrode"])
        G_p = pybamm.interface.homogeneous_reaction(["positive electrode"])
        G = pybamm.interface.homogeneous_reaction(whole_cell)

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
        cn_surf = pybamm.surf(c_n)
        cp_surf = pybamm.surf(c_p)

        # TODO: put in proper expression for cen and cep
        cen = pybamm.Scalar(0)
        cep = pybamm.Scalar(0)
        gn = m_n * cn_surf ** 0.5 * (1 - cn_surf) ** 0.5 * (1 + delta * cen) ** 0.5
        gp = (
            m_p
            * C_hat_p
            * cp_surf ** 0.5
            * (1 - cp_surf) ** 0.5
            * (1 + delta * cep) ** 0.5
        )
        # TODO: put in proper averaging
        gn_av = gn
        gp_av = gp

        # linearise BV for now
        ocp = U_p(cp_surf) - U_n(cn_surf)
        reaction_overpotential = -(2 / Lambda) * (1 / (gp_av * lp)) - (2 / Lambda) * (
            1 / (gn_av * ln)
        )

        # TODO: add the proper expressions for the averages
        cep_av = pybamm.Scalar(0)
        cen_av = pybamm.Scalar(0)
        concentration_overpotential = (
            2 * delta * (1 - t_plus) / Lambda * (cep_av - cen_av)
        )
        electrolyte_ohmic_losses = (
            -delta
            * current
            * nu
            / Lambda
            / sigma_e(c_e)
            * (
                ln / (3 * epsilon_n ** b)
                + ls / (epsilon_s ** b)
                + lp / (3 * epsilon_p ** b)
            )
        )

        electrode_ohmic_losses = -current / 3 * (lp / sigma_p + ln / sigma_n)

        voltage = (
            ocp
            + reaction_overpotential
            + concentration_overpotential
            + electrolyte_ohmic_losses
            + electrode_ohmic_losses
        )
        additional_variables = {
            "cn_surf": cn_surf,
            "cp_surf": cp_surf,
            "voltage": voltage,
        }
        self._variables.update(additional_variables)
