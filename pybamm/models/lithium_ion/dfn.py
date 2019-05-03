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

        "-----------------------------------------------------------------------------"
        "Submodels"
        # Exchange-current density
        c_e_n, _, c_e_p = c_e.orphans
        c_s_n_surf = pybamm.surf(c_s_n, set_domain=True)
        c_s_p_surf = pybamm.surf(c_s_p, set_domain=True)
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j0_n = int_curr_model.get_exchange_current_densities(c_e_n, c_s_n_surf)
        j0_p = int_curr_model.get_exchange_current_densities(c_e_p, c_s_p_surf)

        # Potentials
        phi_e_n, _, phi_e_p = phi_e.orphans
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = phi_s_n - phi_e_n - ocp_n
        eta_r_p = phi_s_p - phi_e_p - ocp_p

        # Interfacial current density
        j_n = int_curr_model.get_butler_volmer(j0_n, eta_r_n)
        j_p = int_curr_model.get_butler_volmer(j0_p, eta_r_p)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p)

        # Electrolyte concentration
        reactions = {
            "main": {"neg": {"s_plus": 1, "aj": j_n}, "pos": {"s_plus": 1, "aj": j_p}}
        }
        # Electrolyte diffusion model
        electrolyte_diffusion_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        electrolyte_diffusion_model.set_differential_system(c_e, reactions)

        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        eleclyte_current_model.set_algebraic_system(phi_e, c_e, reactions)

        # Electrode models
        negative_electrode_current_model = pybamm.electrode.Ohm(param)
        negative_electrode_current_model.set_algebraic_system(phi_s_n, reactions)
        positive_electrode_current_model = pybamm.electrode.Ohm(param)
        positive_electrode_current_model.set_algebraic_system(phi_s_p, reactions)

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

        # Excahnge-current density
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        pot_model = pybamm.potential.Potential(param)
        ocp_vars = pot_model.get_derived_open_circuit_potentials(ocp_n, ocp_p)
        eta_r_vars = pot_model.get_derived_reaction_overpotentials(eta_r_n, eta_r_p)
        self.variables.update({**ocp_vars, **eta_r_vars})

        # Voltage
        phi_s_n = self.variables["Negative electrode potential"]
        phi_s_p = self.variables["Positive electrode potential"]
        i_s_n = self.variables["Negative electrode current density"]
        i_s_p = self.variables["Positive electrode current density"]
        volt_vars = positive_electrode_current_model.get_variables(
            phi_s_n, phi_s_p, i_s_n, i_s_p
        )
        self.variables.update(volt_vars)

        "-----------------------------------------------------------------------------"
        "Defaults and Solver Conditions"
        # default geometry
        self.default_geometry = pybamm.Geometry("1D macro", "1+1D micro")
        # Default solver to DAE
        self.default_solver = pybamm.ScikitsDaeSolver()
