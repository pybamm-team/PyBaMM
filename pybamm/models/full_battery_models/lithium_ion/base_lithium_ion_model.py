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
            "current collector y": self.param.L_y,
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
        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p

        # LAM
        C_n = self.variables["Negative electrode capacity [A.h]"]
        C_p = self.variables["Positive electrode capacity [A.h]"]

        eps_s_n_av_init = pybamm.x_average(param.epsilon_s_n(x_n))
        eps_s_p_av_init = pybamm.x_average(param.epsilon_s_p(x_p))
        C_n_init = (
            eps_s_n_av_init * param.L_n * param.A_cc * param.c_n_max * param.F / 3600
        )
        C_p_init = (
            eps_s_p_av_init * param.L_p * param.A_cc * param.c_p_max * param.F / 3600
        )

        LAM_ne = (1 - C_n / C_n_init) * 100
        LAM_pe = (1 - C_n / C_n_init) * 100

        # LLI
        n_Li_e = self.variables["Total lithium in electrolyte [mol]"]
        n_Li_p = self.variables["Total lithium in positive electrode [mol]"]
        n_Li_n = self.variables["Total lithium in negative electrode [mol]"]
        n_Li_particles = n_Li_n + n_Li_p
        n_Li = n_Li_particles + n_Li_e

        eps_n_init = param.epsilon_n
        eps_s_init = param.epsilon_s
        eps_p_init = param.epsilon_p
        eps_init = pybamm.Concatenation(eps_n_init, eps_s_init, eps_p_init)

        c_e_av_init = pybamm.x_average(eps_init) * param.c_e_typ
        n_Li_e_init = c_e_av_init * param.L_x * param.A_cc

        eps_s_n_init = param.epsilon_s_n(x_n)
        c_n_init = param.c_n_init(x_n)
        c_n_av_init = pybamm.x_average(eps_s_n_init * c_n_init)
        n_Li_n_init = c_n_av_init * param.c_n_max * param.L_n * param.A_cc

        eps_s_p_init = param.epsilon_s_p(x_p)
        c_p_init = param.c_p_init(x_p)
        c_p_av_init = pybamm.x_average(eps_s_p_init * c_p_init)
        n_Li_p_init = c_p_av_init * param.c_p_max * param.L_p * param.A_cc

        n_Li_particles_init = n_Li_n_init + n_Li_p_init
        n_Li_init = n_Li_particles_init + n_Li_e_init

        LLI = (1 - n_Li / n_Li_init) * 100

        self.variables.update(
            {
                "Loss of Active Material in negative electrode [%]": LAM_ne,
                "Loss of Active Material in positive electrode [%]": LAM_pe,
                "Loss of Lithium Inventory [%]": LLI,
                # Total lithium
                "Total lithium [mol]": n_Li,
                "Total lithium in particles [mol]": n_Li_particles,
                "Total lithium in electrolyte [mol]": n_Li_e,
                # Initial total lithium
                "Initial total lithium [mol]": n_Li_init,
                "Initial total lithium in particles [mol]": n_Li_particles_init,
                "Initial total lithium in electrolyte [mol]": n_Li_e_init,
                # Lithium lost
                "Total lithium lost [mol]": n_Li_init - n_Li,
                "Total lithium lost from particles [mol]": n_Li_particles_init
                - n_Li_particles,
                "Total lithium lost from electrolyte [mol]": n_Li_e_init - n_Li_e,
                "eps_n_av_init": pybamm.x_average(eps_s_n_init),
                "c_n_av_init": pybamm.x_average(c_n_init),
            }
        )

        # Lithium lost to side reactions
        # Different way of measuring LLI but should give same value
        LLI_sei_n = self.variables["Loss of lithium to negative electrode sei [mol]"]
        LLI_sei_p = self.variables["Loss of lithium to positive electrode sei [mol]"]

        LLI_reactions = LLI_sei_n + LLI_sei_p
        self.variables.update(
            {"Total lithium lost to side reactions [mol]": LLI_reactions}
        )

    def set_sei_submodel(self):

        # negative electrode SEI
        if self.options["sei"] == "none":
            self.submodels["negative sei"] = pybamm.sei.NoSEI(self.param, "Negative")

        if self.options["sei"] == "constant":
            self.submodels["negative sei"] = pybamm.sei.ConstantSEI(
                self.param, "Negative"
            )

        elif self.options["sei"] == "reaction limited":
            self.submodels["negative sei"] = pybamm.sei.ReactionLimited(
                self.param, "Negative"
            )

        elif self.options["sei"] == "solvent-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.SolventDiffusionLimited(
                self.param, "Negative"
            )

        elif self.options["sei"] == "electron-migration limited":
            self.submodels["negative sei"] = pybamm.sei.ElectronMigrationLimited(
                self.param, "Negative"
            )

        elif self.options["sei"] == "interstitial-diffusion limited":
            self.submodels["negative sei"] = pybamm.sei.InterstitialDiffusionLimited(
                self.param, "Negative"
            )

        elif self.options["sei"] == "ec reaction limited":
            self.submodels["negative sei"] = pybamm.sei.EcReactionLimited(
                self.param, "Negative"
            )

        # positive electrode
        self.submodels["positive sei"] = pybamm.sei.NoSEI(self.param, "Positive")

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
        elif self.options["particle cracking"] == "cathode":
            n = pybamm.particle_cracking.NoCracking(self.param, "Negative")
            p = pybamm.particle_cracking.CrackPropagation(self.param, "Positive")
        elif self.options["particle cracking"] == "anode":
            n = pybamm.particle_cracking.CrackPropagation(self.param, "Negative")
            p = pybamm.particle_cracking.NoCracking(self.param, "Positive")
        else:
            n = pybamm.particle_cracking.CrackPropagation(self.param, "Negative")
            p = pybamm.particle_cracking.CrackPropagation(self.param, "Positive")

        self.submodels["negative particle cracking"] = n
        self.submodels["positive particle cracking"] = p
