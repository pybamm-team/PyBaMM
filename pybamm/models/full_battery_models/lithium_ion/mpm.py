#
# Many-Particle Model (MPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class MPM(BaseModel):
    """Many-Particle Model (MPM) of a lithium-ion battery with particle-size
    distributions for each electrode, from [1]_.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).

    References
    ----------
    .. [1] TL Kirk, J Evans, CP Please and SJ Chapman. “Modelling electrode
        heterogeneity in lithium-ion batteries: unimodal and bimodal particle-size
        distributions”.
        In: arXiv preprint arXiv:2006.12208 (2020).

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self, options=None, name="Many-Particle Model", build=True
    ):
        super().__init__(options, name)
        self.options["particle size"] = "distribution"

        # Set submodels
        self.set_external_circuit_submodel()
        self.set_porosity_submodel()
        self.set_crack_submodel()
        self.set_active_material_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_interfacial_submodel()
        self.set_other_reaction_submodels_to_zero()
        self.set_particle_submodel()
        self.set_negative_electrode_submodel()
        self.set_electrolyte_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_current_collector_submodel()

        self.set_sei_submodel()
        self.set_lithium_plating_submodel()

        if build:
            self.build_model()

        # pybamm.citations.register("marquis2019asymptotic")

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
            self.submodels["porosity"] = pybamm.porosity.LeadingOrder(
                self.param, self.options
            )

    def set_active_material_submodel(self):

        if self.options["loss of active material"] == "none":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.Constant(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.Constant(self.param, "Positive", self.options)
        elif self.options["loss of active material"] == "both":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.VaryingUniform(
                self.param, "Negative", self.options
            )
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.VaryingUniform(
                self.param, "Positive", self.options
            )
        elif self.options["loss of active material"] == "negative":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.VaryingUniform(
                self.param, "Negative", self.options
            )
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.Constant(self.param, "Positive", self.options)
        elif self.options["loss of active material"] == "positive":
            self.submodels[
                "negative active material"
            ] = pybamm.active_material.Constant(self.param, "Negative", self.options)
            self.submodels[
                "positive active material"
            ] = pybamm.active_material.VaryingUniform(
                self.param, "Positive", self.options
            )

    def set_convection_submodel(self):

        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param)
        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param)

    def set_interfacial_submodel(self):

        self.submodels["negative interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Negative", "lithium-ion main", self.options
        )

        self.submodels["positive interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Positive", "lithium-ion main", self.options
        )

    def set_particle_submodel(self):

        if self.options["particle"] == "Fickian diffusion":
            submod_n = pybamm.particle.FickianSingleSizeDistribution(
                self.param, "Negative"
            )
            submod_p = pybamm.particle.FickianSingleSizeDistribution(
                self.param, "Positive"
            )
        elif self.options["particle"] == "uniform profile":
            submod_n = pybamm.particle.FastSingleSizeDistribution(
                self.param, "Negative"
            )
            submod_p = pybamm.particle.FastSingleSizeDistribution(
                self.param, "Positive"
            )
        self.submodels["negative particle"] = submod_n
        self.submodels["positive particle"] = submod_p

    def set_negative_electrode_submodel(self):

        self.submodels[
            "negative electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrderSizeDistribution(self.param, "Negative")

    def set_positive_electrode_submodel(self):

        self.submodels[
            "positive electrode potential"
        ] = pybamm.electrode.ohm.LeadingOrderSizeDistribution(self.param, "Positive")

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["electrolyte conductivity"] not in ["default", "leading order"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for SPM".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if self.options["surface form"] == "false":
            self.submodels[
                "leading-order electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.LeadingOrder(self.param)

        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    "leading-order " + domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderDifferential(self.param, domain)

        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    "leading-order " + domain.lower() + " electrolyte conductivity"
                ] = surf_form.LeadingOrderAlgebraic(self.param, domain)

        self.submodels[
            "electrolyte diffusion"
        ] = pybamm.electrolyte_diffusion.ConstantConcentration(self.param)

    @property
    def default_parameter_values(self):
        # Default parameter values
        default_params = super().default_parameter_values
        # Extract the particle radius, taken to be the average radius
        R_n_dim = default_params["Negative particle radius [m]"]
        R_p_dim = default_params["Positive particle radius [m]"]

        # Additional particle distribution parameter values

        # Area-weighted standard deviations
        sd_a_n = 0.3
        sd_a_p = 0.3

        # Minimum radius in the particle-size distributions (dimensionless).
        R_min_n = 0
        R_min_p = 0

        # Max radius in the particle-size distributions (dimensionless).
        # Either 5 s.d.'s above the mean or 2 times the mean, whichever is larger
        R_max_n = max(2, 1 + sd_a_n * 5)
        R_max_p = max(2, 1 + sd_a_p * 5)

        # Define lognormal distribution
        def lognormal_distribution(R, R_av, sd):
            '''
            A lognormal distribution with arguments
                R :     particle radius
                R_av:   mean particle radius
                sd :    standard deviation
            (Inputs can be dimensional or dimensionless)
            '''
            import numpy as np

            mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
            sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
            return (
                pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
                / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
                / (R)
            )

        # Set the (area-weighted) particle-size distributions (dimensional)
        def f_a_dist_n_dim(R):
            return lognormal_distribution(R, R_n_dim, sd_a_n * R_n_dim)

        def f_a_dist_p_dim(R):
            return lognormal_distribution(R, R_p_dim, sd_a_p * R_p_dim)

        # Append to default parameters (dimensional)
        default_params.update(
            {
                "Negative area-weighted particle-size "
                + "standard deviation [m]": sd_a_n * R_n_dim,
                "Positive area-weighted particle-size "
                + "standard deviation [m]": sd_a_p * R_p_dim,
                "Negative minimum particle radius [m]": R_min_n * R_n_dim,
                "Positive minimum particle radius [m]": R_min_p * R_p_dim,
                "Negative maximum particle radius [m]": R_max_n * R_n_dim,
                "Positive maximum particle radius [m]": R_max_p * R_p_dim,
                "Negative area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_n_dim,
                "Positive area-weighted "
                + "particle-size distribution [m-1]": f_a_dist_p_dim,
            },
            check_already_exists=False,
        )
        return default_params


