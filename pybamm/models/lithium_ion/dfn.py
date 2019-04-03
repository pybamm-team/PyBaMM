#
# Doyle-Fuller-Newman (DFN) Model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


class DFN(pybamm.LithiumIonBaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "Parameters"
        param = pybamm.standard_parameters
        param.__dict__.update(pybamm.standard_parameters_lithium_ion.__dict__)

        "Model Variables"
        # Electrolyte concentration
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration", ["negative electrode"]
        )
        c_e_s = pybamm.Variable("Separator electrolyte concentration", ["separator"])
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration", ["positive electrode"]
        )
        c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte Potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential", ["negative electrode"]
        )
        phi_e_s = pybamm.Variable("Separator electrolyte potential", ["separator"])
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential", ["positive electrode"]
        )
        phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode Potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential", ["negative electrode"]
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential", ["positive electrode"]
        )

        # Particle concentration
        c_s_n = pybamm.Variable(
            "Negative particle concentration", ["negative particle"]
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration", ["positive particle"]
        )

        "Submodels"
        # Interfacial current density
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        j_n = pybamm.interface.butler_volmer(
            param, c_e_n, phi_s_n - phi_e_n, c_s_k_surf=c_s_n_surf
        )
        j_s = pybamm.Broadcast(0, ["separator"])
        j_p = pybamm.interface.butler_volmer(
            param, c_e_p, phi_s_p - phi_e_p, c_s_k_surf=c_s_p_surf
        )
        j = pybamm.Concatenation(j_n, j_s, j_p)

        # Electrolyte models
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(
            c_e, j, param
        )
        electrolyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(
            c_e, phi_e, j, param
        )

        # Electrode models
        negative_electrode_current_model = pybamm.electrode.Ohm(phi_s_n, j_n, param)
        positive_electrode_current_model = pybamm.electrode.Ohm(phi_s_p, j_p, param)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(c_s_n, j_n, param)
        positive_particle_model = pybamm.particle.Standard(c_s_p, j_p, param)

        "Combine Submodels"
        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
            electrolyte_current_model,
            negative_electrode_current_model,
            positive_electrode_current_model,
        )

        "Additional Conditions"
        additional_bcs = {}
        self._boundary_conditions.update(additional_bcs)

        "Additional Model Variables"
        additional_variables = {}
        self._variables.update(additional_variables)

        "Solver Conditions"
        # Default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
        # Cut-off if either concentration goes negative
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
