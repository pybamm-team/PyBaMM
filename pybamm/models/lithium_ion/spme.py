#
# Single Particle Model with Electrolyte (SPMe)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class SPMe(pybamm.LithiumIonBaseModel):
    """Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self):
        super().__init__()

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = pybamm.standard_parameters_lithium_ion

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Interfacial current density
        interfacial_current_model = pybamm.interface.InterfacialCurrent(param)
        interfacial_current_model.set_homogeneous_interfacial_current()
        self.update(interfacial_current_model)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(
            self.variables, ["negative particle"]
        )
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(
            self.variables, ["positive particle"]
        )

        # Electrolyte model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(self.variables)

        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Exchange-current density
        interfacial_current_model.set_exchange_current_densities(self.variables)
        self.update(interfacial_current_model)

        # Potentials
        potential_model = pybamm.potential.Potential(param)
        c_s_n_surf = self.variables["Negative particle surface concentration"]
        c_s_p_surf = self.variables["Positive particle surface concentration"]
        potential_model.set_open_circuit_potentials(c_s_n_surf, c_s_p_surf)
        potential_model.set_reaction_overpotentials(self.variables, "current")
        self.update(potential_model)

        # Electrolyte current
        # Define leading-order concentration
        c_e_0 = pybamm.Scalar(1)
        self.variables.update({"Electrolyte concentration (leading-order)": c_e_0})
        # Call model
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_explicit_combined(self.variables)
        self.update(eleclyte_current_model)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_model.set_explicit_combined(self.variables)
        self.update(electrode_model)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")

        # Cut-off if either concentration goes negative
        c_s_n = self.variables["Negative particle concentration"]
        c_s_p = self.variables["Positive particle concentration"]
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
