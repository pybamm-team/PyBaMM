#
# Lithium-ion base model class
#
import pybamm


class BaseModel(pybamm.BaseBatteryModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lithium-ion models

    **Extends:** :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None, name="Unnamed lithium-ion model", build=False):
        super().__init__(options, name)
        self.param = pybamm.LithiumIonParameters(options)

        # Assess whether the submodel is a half-cell model
        self.half_cell = self.options["working electrode"] != "both"

        # Default timescale is discharge timescale
        self.timescale = self.param.tau_discharge

        # Set default length scales
        self.length_scales = {
            "negative electrode": self.param.L_x,
            "separator": self.param.L_x,
            "positive electrode": self.param.L_x,
            "positive particle": self.param.R_p_typ,
            "positive particle size": self.param.R_p_typ,
            "current collector y": self.param.L_z,
            "current collector z": self.param.L_z,
        }

        # Add negative particle domains only if not a half cell model
        if not self.half_cell:
            self.length_scales.update(
                {
                    "negative particle": self.param.R_n_typ,
                    "negative particle size": self.param.R_n_typ,
                }
            )
        self.set_standard_output_variables()

    def set_standard_output_variables(self):
        super().set_standard_output_variables()

        # Particle concentration position
        var = pybamm.standard_spatial_vars
        self.variables.update({"r_p": var.r_p, "r_p [m]": var.r_p * self.param.R_p_typ})
        if not self.half_cell:
            self.variables.update(
                {"r_n": var.r_n, "r_n [m]": var.r_n * self.param.R_n_typ}
            )

    def set_degradation_variables(self):
        """Sets variables that quantify degradation (LAM, LLI, etc)"""
        param = self.param

        # LAM
        if self.half_cell:
            n_Li_n = pybamm.Scalar(0)
            LAM_ne = pybamm.Scalar(0)
        else:
            C_n = self.variables["Negative electrode capacity [A.h]"]
            n_Li_n = self.variables["Total lithium in negative electrode [mol]"]
            LAM_ne = (1 - C_n / param.C_n_init) * 100

        C_p = self.variables["Positive electrode capacity [A.h]"]

        LAM_pe = (1 - C_p / param.C_p_init) * 100

        # LLI
        n_Li_e = self.variables["Total lithium in electrolyte [mol]"]
        n_Li_p = self.variables["Total lithium in positive electrode [mol]"]
        n_Li_particles = n_Li_n + n_Li_p
        n_Li = n_Li_particles + n_Li_e

        # LLI is usually defined based only on the percentage lithium lost from
        # particles
        LLI = (1 - n_Li_particles / param.n_Li_particles_init) * 100
        LLI_tot = (1 - n_Li / param.n_Li_init) * 100

        self.variables.update(
            {
                "LAM_ne [%]": LAM_ne,
                "LAM_pe [%]": LAM_pe,
                "LLI [%]": LLI,
                "Loss of active material in negative electrode [%]": LAM_ne,
                "Loss of active material in positive electrode [%]": LAM_pe,
                "Loss of lithium inventory [%]": LLI,
                "Loss of lithium inventory, including electrolyte [%]": LLI_tot,
                # Total lithium
                "Total lithium [mol]": n_Li,
                "Total lithium in particles [mol]": n_Li_particles,
                # Lithium lost
                "Total lithium lost [mol]": param.n_Li_init - n_Li,
                "Total lithium lost from particles [mol]": param.n_Li_particles_init
                - n_Li_particles,
                "Total lithium lost from electrolyte [mol]": param.n_Li_e_init - n_Li_e,
            }
        )

        # Lithium lost to side reactions
        # Different way of measuring LLI but should give same value
        LLI_sei = self.variables["Loss of lithium to SEI [mol]"]
        LLI_pl = self.variables["Loss of lithium to lithium plating [mol]"]

        LLI_reactions = LLI_sei + LLI_pl
        self.variables.update(
            {
                "Total lithium lost to side reactions [mol]": LLI_reactions,
                "Total capacity lost to side reactions [A.h]": LLI_reactions
                * param.F
                / 3600,
            }
        )

    def set_sei_submodel(self):

        # SEI
        if self.options["SEI"] == "none":
            self.submodels["sei"] = pybamm.sei.NoSEI(self.param, self.options)

        if self.options["SEI"] == "constant":
            self.submodels["sei"] = pybamm.sei.ConstantSEI(self.param)

        elif self.options["SEI"] == "reaction limited":
            self.submodels["sei"] = pybamm.sei.ReactionLimited(
                self.param, self.x_average
            )

        elif self.options["SEI"] == "solvent-diffusion limited":
            self.submodels["sei"] = pybamm.sei.SolventDiffusionLimited(
                self.param, self.x_average
            )

        elif self.options["SEI"] == "electron-migration limited":
            self.submodels["sei"] = pybamm.sei.ElectronMigrationLimited(
                self.param, self.x_average
            )

        elif self.options["SEI"] == "interstitial-diffusion limited":
            self.submodels["sei"] = pybamm.sei.InterstitialDiffusionLimited(
                self.param, self.x_average
            )

        elif self.options["SEI"] == "ec reaction limited":
            self.submodels["sei"] = pybamm.sei.EcReactionLimited(
                self.param, self.x_average
            )

    def set_lithium_plating_submodel(self):

        # negative electrode
        if self.options["lithium plating"] == "none":
            self.submodels["lithium plating"] = pybamm.lithium_plating.NoPlating(
                self.param, self.options
            )

        elif self.options["lithium plating"] == "reversible":
            self.submodels[
                "lithium plating"
            ] = pybamm.lithium_plating.ReversiblePlating(self.param, self.x_average)

        elif self.options["lithium plating"] == "irreversible":
            self.submodels[
                "lithium plating"
            ] = pybamm.lithium_plating.IrreversiblePlating(self.param, self.x_average)

    def set_other_reaction_submodels_to_zero(self):
        self.submodels["negative oxygen interface"] = pybamm.interface.NoReaction(
            self.param, "Negative", "lithium-ion oxygen"
        )
        self.submodels["positive oxygen interface"] = pybamm.interface.NoReaction(
            self.param, "Positive", "lithium-ion oxygen"
        )

    def set_crack_submodel(self):
        # this option can either be a string (both sides the same) or a 2-tuple
        # to indicate different options in negative and positive electrodes
        if isinstance(self.options["particle mechanics"], str):
            crack_left = self.options["particle mechanics"]
            crack_right = self.options["particle mechanics"]
        else:
            crack_left, crack_right = self.options["particle mechanics"]
        for crack_side, domain in [[crack_left, "Negative"], [crack_right, "Positive"]]:
            if crack_side == "none":
                pass
            elif crack_side == "swelling only":
                self.submodels[
                    domain.lower() + " particle mechanics"
                ] = pybamm.particle_mechanics.SwellingOnly(self.param, domain)
            elif crack_side == "swelling and cracking":
                self.submodels[
                    domain.lower() + " particle mechanics"
                ] = pybamm.particle_mechanics.CrackPropagation(
                    self.param, domain, self.x_average
                )

    def set_active_material_submodel(self):
        # this option can either be a string (both sides the same) or a 2-tuple
        # to indicate different options in negative and positive electrodes
        if isinstance(self.options["loss of active material"], str):
            lam_left = self.options["loss of active material"]
            lam_right = self.options["loss of active material"]
        else:
            lam_left, lam_right = self.options["loss of active material"]
        for lam_side, domain in [[lam_left, "Negative"], [lam_right, "Positive"]]:
            if lam_side == "none":
                self.submodels[
                    domain.lower() + " active material"
                ] = pybamm.active_material.Constant(self.param, domain, self.options)
            elif lam_side == "stress-driven":
                self.submodels[
                    domain.lower() + " active material"
                ] = pybamm.active_material.StressDriven(
                    self.param, domain, self.options, self.x_average
                )
            elif lam_side == "reaction-driven":
                self.submodels[
                    domain.lower() + " active material"
                ] = pybamm.active_material.ReactionDriven(
                    self.param, domain, self.options, self.x_average
                )

    def set_porosity_submodel(self):
        if (
            self.options["SEI porosity change"] == "false"
            and self.options["lithium plating porosity change"] == "false"
        ):
            self.submodels["porosity"] = pybamm.porosity.Constant(
                self.param, self.options
            )
        elif (
            self.options["SEI porosity change"] == "true"
            or self.options["lithium plating porosity change"] == "true"
        ):
            self.submodels["porosity"] = pybamm.porosity.ReactionDriven(
                self.param, self.options, self.x_average
            )
