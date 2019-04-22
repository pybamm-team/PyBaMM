#
# Doyle-Fuller-Newman (DFN) Model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class DFN(pybamm.LithiumIonBaseModel):
    """Doyle-Fuller-Newman (DFN) model of a lithium-ion battery.
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
        phi_e = pybamm.standard_variables.phi_e
        phi_s_p = pybamm.standard_variables.phi_s_p
        phi_s_n = pybamm.standard_variables.phi_s_n

        # Add variables to list of variables, as they are needed by submodels
        self.variables.update(
            {
                "Negative particle concentration": c_s_n,
                "Positive particle concentration": c_s_p,
                "Electrolyte concentration": c_e,
                "Electrolyte potential": phi_e,
                "Negative electrode potential": phi_s_n,
                "Positive electrode potential": phi_s_p,
            }
        )

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Exchange-current density
        int_curr_model = pybamm.interface.InterfacialCurrent(param)
        ecd_vars = int_curr_model.get_exchange_current_densities(self.variables)
        self.variables.update(ecd_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_n_surf.domain = ["negative electrode"]
        c_s_p_surf = pybamm.surf(c_s_p)
        c_s_p_surf.domain = ["positive electrode"]
        ocp_vars = pot_model.get_open_circuit_potentials(c_s_n_surf, c_s_p_surf)
        self.variables.update(ocp_vars)
        eta_r_vars = pot_model.get_reaction_overpotentials(self.variables, "potentials")
        self.variables.update(eta_r_vars)

        # Interfacial current density
        j_vars = int_curr_model.get_interfacial_current_butler_volmer(self.variables)
        self.variables.update(j_vars)

        # Particle models
        j_n = j_vars["Negative electrode interfacial current density"]
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n)
        j_p = j_vars["Positive electrode interfacial current density"]
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p)

        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(c_e, self.variables)

        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_algebraic_system(phi_e, self.variables)

        # Electrode models
        negative_electrode_current_model = pybamm.electrode.Ohm(param)
        negative_electrode_current_model.set_algebraic_system(phi_s_n, self.variables)
        positive_electrode_current_model = pybamm.electrode.Ohm(param)
        positive_electrode_current_model.set_algebraic_system(phi_s_p, self.variables)

        "-----------------------------------------------------------------------------"
        "Combine Submodels"
        self.update(
            negative_particle_model,
            positive_particle_model,
            electrolyte_diffusion_model,
            eleclyte_current_model,
            negative_electrode_current_model,
            positive_electrode_current_model,
        )

        "-----------------------------------------------------------------------------"
        "Post-process"
        volt_vars = positive_electrode_current_model.get_post_processed(self.variables)
        self.variables.update(volt_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1+1D micro")
        # Default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
