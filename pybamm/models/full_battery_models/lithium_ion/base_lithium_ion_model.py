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

        # Default timescale is discharge timescale
        self.timescale = self.param.tau_discharge

        # Set default length scales
        self.length_scales = {
            "negative electrode": self.param.L_x,
            "separator": self.param.L_x,
            "positive electrode": self.param.L_x,
            "negative particle": self.param.R_n_typ,
            "positive particle": self.param.R_p_typ,
            "current collector y": self.param.L_z,
            "current collector z": self.param.L_z,
        }
        self.set_standard_output_variables()

    def set_standard_output_variables(self):
        super().set_standard_output_variables()

        # Particle concentration position
        var = pybamm.standard_spatial_vars
        self.variables.update(
            {
                "r_n": var.r_n,
                "r_n [m]": var.r_n * self.param.R_n_typ,
                "r_p": var.r_p,
                "r_p [m]": var.r_p * self.param.R_p_typ,
            }
        )

    def set_sei_submodel(self):

        # negative electrode SEI
        if self.options["SEI"] == "none":
            self.submodels["negative sei"] = pybamm.sei.NoSEI(self.param, "Negative")

        if self.options["SEI"] == "constant":
            self.submodels["negative sei"] = pybamm.sei.ConstantSEI(
                self.param, "Negative"
            )

        elif self.options["SEI"] == "reaction limited":
            self.submodels["negative sei"] = pybamm.sei.ReactionLimited(
                self.param, "Negative", self.x_average
            )

        elif self.options["SEI"] == "solvent-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.SolventDiffusionLimited(
                self.param, "Negative", self.x_average
            )

        elif self.options["SEI"] == "electron-migration limited":
            self.submodels["negative sei"] = pybamm.sei.ElectronMigrationLimited(
                self.param, "Negative", self.x_average
            )

        elif self.options["SEI"] == "interstitial-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.InterstitialDiffusionLimited(
                self.param, "Negative", self.x_average
            )

        elif self.options["SEI"] == "ec reaction limited":
            self.submodels["negative sei"] = pybamm.sei.EcReactionLimited(
                self.param, "Negative", self.x_average
            )

        # positive electrode
        self.submodels["positive sei"] = pybamm.sei.NoSEI(self.param, "Positive")

    def set_lithium_plating_submodel(self):

        # negative electrode
        if self.options["lithium plating"] == "none":
            self.submodels[
                "negative lithium plating"
            ] = pybamm.lithium_plating.NoPlating(self.param, "Negative")

        elif self.options["lithium plating"] == "reversible":
            self.submodels[
                "negative lithium plating"
            ] = pybamm.lithium_plating.ReversiblePlating(
                self.param, "Negative", self.x_average
            )

        elif self.options["lithium plating"] == "irreversible":
            self.submodels[
                "negative lithium plating"
            ] = pybamm.lithium_plating.IrreversiblePlating(
                self.param, "Negative", self.x_average
            )

        # positive electrode
        self.submodels["positive lithium plating"] = pybamm.lithium_plating.NoPlating(
            self.param, "Positive"
        )

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
                ] = pybamm.particle_cracking.SwellingOnly(self.param, domain)
            elif crack_side == "swelling and cracking":
                self.submodels[
                    domain.lower() + " particle mechanics"
                ] = pybamm.particle_cracking.CrackPropagation(
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
