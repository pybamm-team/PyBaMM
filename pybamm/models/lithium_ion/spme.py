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
        "Model Variables"

        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p
        c_e = pybamm.standard_variables.c_e

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Interfacial current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        j_vars = int_curr_model.get_homogeneous_interfacial_current()
        self.variables.update(j_vars)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, self.variables)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, self.variables)
        self.update(negative_particle_model, positive_particle_model)

        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(c_e, self.variables)
        self.update(electrolyte_diffusion_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"
        # Exchange-current density
        ecd_vars = int_curr_model.get_exchange_current_densities(self.variables)
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_open_circuit_potentials(
            self.variables, intercalation=True
        )
        eta_r_vars = pot_model.get_reaction_overpotentials(self.variables, "current")
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Electrolyte current
        # Define leading-order concentration
        c_e_0 = pybamm.Scalar(1)
        self.variables.update({"Electrolyte concentration (leading-order)": c_e_0})
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_combined(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_combined(self.variables)
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1D micro")

        # Cut-off if either concentration goes negative
        c_s_n = self.variables["Negative particle concentration"]
        c_s_p = self.variables["Positive particle concentration"]
        self.events = [pybamm.Function(np.min, c_s_n), pybamm.Function(np.min, c_s_p)]
