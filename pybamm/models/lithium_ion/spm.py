#
# Single Particle Model (SPM)
#
import pybamm


class SPM(pybamm.LithiumIonBaseModel):
    """Single Particle Model (SPM) of a lithium-ion battery.
    **Extends:** :class:`pybamm.LithiumIonBaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Single Particle Model"

        "-----------------------------------------------------------------------------"
        "Parameters"
        param = self.set_of_parameters

        "-----------------------------------------------------------------------------"
        "Model Variables"
        c_s_n = pybamm.standard_variables.c_s_n
        c_s_p = pybamm.standard_variables.c_s_p

        if self.options["bc_options"]["dimensionality"] == 0:
            i_boundary_cc = param.current_with_time
            self.variables["Current collector current density"] = i_boundary_cc
            curr_coll_domain = []
            broadcast = True
        elif self.options["bc_options"]["dimensionality"] == 1:
            raise NotImplementedError
        elif self.options["bc_options"]["dimensionality"] == 2:
            i_boundary_cc = pybamm.Variable(
                "Current collector current density", domain="current collector"
            )
            self.variables["Current collector current density"] = i_boundary_cc
            curr_coll_domain = ["current collector"]
            broadcast = False

        "-----------------------------------------------------------------------------"
        "Submodels"

        # Interfacial current density
        neg = ["negative electrode"]
        pos = ["positive electrode"]
        int_curr_model = pybamm.interface.LithiumIonReaction(param)
        j_n = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, neg)
        j_p = int_curr_model.get_homogeneous_interfacial_current(i_boundary_cc, pos)

        # Particle models
        negative_particle_model = pybamm.particle.Standard(param)
        negative_particle_model.set_differential_system(c_s_n, j_n, broadcast=broadcast)
        positive_particle_model = pybamm.particle.Standard(param)
        positive_particle_model.set_differential_system(c_s_p, j_p, broadcast=broadcast)
        self.update(negative_particle_model, positive_particle_model)

        "-----------------------------------------------------------------------------"
        "Post-Processing"

        # Electrolyte concentration
        c_e = pybamm.Scalar(1)
        N_e = pybamm.Scalar(0)
        electrolyte_conc_model = pybamm.electrolyte_diffusion.StefanMaxwell(param)
        conc_vars = electrolyte_conc_model.get_variables(c_e, N_e)
        self.variables.update(conc_vars)

        # Exchange-current density
        c_s_n_surf = pybamm.surf(c_s_n)
        c_s_p_surf = pybamm.surf(c_s_p)
        c_s_n_surf.domain = curr_coll_domain
        c_s_p_surf.domain = curr_coll_domain

        j0_n = int_curr_model.get_exchange_current_densities(c_e, c_s_n_surf, neg)
        j0_p = int_curr_model.get_exchange_current_densities(c_e, c_s_p_surf, pos)
        j_vars = int_curr_model.get_derived_interfacial_currents(j_n, j_p, j0_n, j0_p)
        self.variables.update(j_vars)

        # Potentials
        ocp_n = param.U_n(c_s_n_surf)
        ocp_p = param.U_p(c_s_p_surf)
        eta_r_n = int_curr_model.get_inverse_butler_volmer(j_n, j0_n, neg)
        eta_r_p = int_curr_model.get_inverse_butler_volmer(j_p, j0_p, pos)
        pot_model = pybamm.potential.Potential(param)
        pot_vars = pot_model.get_all_potentials(
            (ocp_n, ocp_p), eta_r=(eta_r_n, eta_r_p)
        )
        self.variables.update(pot_vars)

        # Electrolyte current
        eleclyte_current_model = pybamm.electrolyte_current.MacInnesStefanMaxwell(param)
        elyte_vars = eleclyte_current_model.get_explicit_leading_order(self.variables)
        self.variables.update(elyte_vars)

        # Electrode
        electrode_model = pybamm.electrode.Ohm(param)
        electrode_vars = electrode_model.get_explicit_leading_order(self.variables)
        self.variables.update(electrode_vars)

        "-----------------------------------------------------------------------------"
        "Boundary conditions"
        if self.options["bc_options"]["dimensionality"] == 2:
            current_collector_model = pybamm.current_collector.OhmTwoDimensional(param)
            # current_collector_model.set_uniform_current(self.variables)
            current_collector_model.set_potential_pair_spm(self.variables)
            self.update(current_collector_model)

        "-----------------------------------------------------------------------------"
        "Events"
        # Cut-off voltage
        # TO DO: get terminal voltage in 2D
        if self.options["bc_options"]["dimensionality"] == 0:
            voltage = self.variables["Terminal voltage"]
            self.events["Minimum voltage cut-off"] = voltage - param.voltage_low_cut
        elif self.options["bc_options"]["dimensionality"] == 2:
            phi_s_cn = self.variables["Negative current collector potential"]
            phi_s_cp = self.variables["Positive current collector potential"]
            voltage = phi_s_cp - phi_s_cn
            self.events["Minimum voltage cut-off"] = (
                pybamm.min(voltage) - param.voltage_low_cut
            )

    @property
    def default_geometry(self):
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            return pybamm.Geometry("1D macro", "1D micro")
        elif dimensionality == 1:
            return pybamm.Geometry("1+1D macro", "(1+0)+1D micro")
        elif dimensionality == 2:
            return pybamm.Geometry("2+1D macro", "(2+0)+1D micro")

    @property
    def default_submesh_types(self):
        base_submeshes = {
            "negative electrode": pybamm.Uniform1DSubMesh,
            "separator": pybamm.Uniform1DSubMesh,
            "positive electrode": pybamm.Uniform1DSubMesh,
            "negative particle": pybamm.Uniform1DSubMesh,
            "positive particle": pybamm.Uniform1DSubMesh,
            "current collector": pybamm.Uniform1DSubMesh,
        }
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality in [0, 1]:
            return base_submeshes
        elif dimensionality == 2:
            base_submeshes["current collector"] = pybamm.Scikit2DSubMesh
            return base_submeshes

    @property
    def default_spatial_methods(self):
        base_spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
            "current collector": pybamm.FiniteVolume,
        }
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality in [0, 1]:
            return base_spatial_methods
        elif dimensionality == 2:
            base_spatial_methods["current collector"] = pybamm.ScikitFiniteElement
            return base_spatial_methods

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        dimensionality = self.options["bc_options"]["dimensionality"]
        if dimensionality == 0:
            return pybamm.ScipySolver()
        else:
            return pybamm.ScikitsDaeSolver()
