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

    def set_degradation_variables(self):
        """ Sets variables that quantify degradation (LAM, LLI, etc) """
        param = self.param

        # LAM
        C_n = self.variables["Negative electrode capacity [A.h]"]
        C_p = self.variables["Positive electrode capacity [A.h]"]

        LAM_ne = (1 - C_n / param.C_n_init) * 100
        LAM_pe = (1 - C_p / param.C_p_init) * 100

        # LLI
        n_Li_e = self.variables["Total lithium in electrolyte [mol]"]
        n_Li_p = self.variables["Total lithium in positive electrode [mol]"]
        n_Li_n = self.variables["Total lithium in negative electrode [mol]"]
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
        LLI_sei_n = self.variables["Loss of lithium to negative electrode SEI [mol]"]
        LLI_sei_p = self.variables["Loss of lithium to positive electrode SEI [mol]"]
        LLI_pl_n = self.variables[
            "Loss of lithium to negative electrode lithium plating [mol]"
        ]
        LLI_pl_p = self.variables[
            "Loss of lithium to positive electrode lithium plating [mol]"
        ]

        LLI_reactions = LLI_sei_n + LLI_sei_p + LLI_pl_n + LLI_pl_p
        self.variables.update(
            {
                "Total lithium lost to side reactions [mol]": LLI_reactions,
                "Total capacity lost to side reactions [A.h]": LLI_reactions
                * param.F
                / 3600,
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
                self.param, "Negative"
            )

        elif self.options["SEI"] == "solvent-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.SolventDiffusionLimited(
                self.param, "Negative"
            )

        elif self.options["SEI"] == "electron-migration limited":
            self.submodels["negative sei"] = pybamm.sei.ElectronMigrationLimited(
                self.param, "Negative"
            )

        elif self.options["SEI"] == "interstitial-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.InterstitialDiffusionLimited(
                self.param, "Negative"
            )

        elif self.options["SEI"] == "ec reaction limited":
            self.submodels["negative sei"] = pybamm.sei.EcReactionLimited(
                self.param, "Negative"
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
            ] = pybamm.lithium_plating.ReversiblePlating(self.param, "Negative")

        elif self.options["lithium plating"] == "irreversible":
            self.submodels[
                "negative lithium plating"
            ] = pybamm.lithium_plating.IrreversiblePlating(self.param, "Negative")

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
        if self.options["particle cracking"] == "none":
            return

        if self.options["particle cracking"] == "no cracking":
            n = pybamm.particle_cracking.NoCracking(self.param, "Negative")
            p = pybamm.particle_cracking.NoCracking(self.param, "Positive")
        elif self.options["particle cracking"] == "positive":
            n = pybamm.particle_cracking.NoCracking(self.param, "Negative")
            p = pybamm.particle_cracking.CrackPropagation(self.param, "Positive")
        elif self.options["particle cracking"] == "negative":
            n = pybamm.particle_cracking.CrackPropagation(self.param, "Negative")
            p = pybamm.particle_cracking.NoCracking(self.param, "Positive")
        else:
            n = pybamm.particle_cracking.CrackPropagation(self.param, "Negative")
            p = pybamm.particle_cracking.CrackPropagation(self.param, "Positive")

        self.submodels["negative particle cracking"] = n
        self.submodels["positive particle cracking"] = p
