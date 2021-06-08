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
        if options is None:
            options = {"particle size": "distribution"}
        else:
            options["particle size"] = "distribution"
        super().__init__(options, name)

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

        pybamm.citations.register("Kirk2020")

    def set_external_circuit_submodel(self):
        """
        Define how the external circuit defines the boundary conditions for the model,
        e.g. (not necessarily constant-) current, voltage, etc
        """
        if self.options["operating mode"] == "current":
            self.submodels["external circuit"] = pybamm.external_circuit.CurrentControl(
                self.param
            )
        elif self.options["operating mode"] == "voltage":
            raise NotImplementedError(
                """Many-Particle Model does not support voltage control."""
            )
        elif self.options["operating mode"] == "power":
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.PowerFunctionControl(self.param)
        elif callable(self.options["operating mode"]):
            self.submodels[
                "external circuit"
            ] = pybamm.external_circuit.FunctionControl(
                self.param, self.options["operating mode"]
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

    def set_thermal_submodel(self):

        if self.options["thermal"] == "isothermal":
            thermal_submodel = pybamm.thermal.isothermal.Isothermal(self.param)

        elif self.options["thermal"] == "lumped":
            thermal_submodel = pybamm.thermal.Lumped(
                self.param,
                cc_dimension=self.options["dimensionality"],
                geometry=self.options["cell geometry"],
            )

        elif self.options["thermal"] == "x-lumped":
            if self.options["dimensionality"] == 0:
                # With 0D current collectors x-lumped is equivalent to lumped pouch
                thermal_submodel = pybamm.thermal.Lumped(self.param, geometry="pouch")
            elif self.options["dimensionality"] == 1:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector1D(
                    self.param
                )
            elif self.options["dimensionality"] == 2:
                thermal_submodel = pybamm.thermal.pouch_cell.CurrentCollector2D(
                    self.param
                )

        elif self.options["thermal"] == "x-full":
            raise NotImplementedError(
                """X-full thermal submodels do
                not yet support particle-size distributions."""
            )

        self.submodels["thermal"] = thermal_submodel

    def set_sei_submodel(self):

        # negative electrode SEI
        if self.options["SEI"] == "none":
            self.submodels["negative sei"] = pybamm.sei.NoSEI(self.param, "Negative")
        else:
            raise NotImplementedError(
                """SEI submodels do not yet support particle-size distributions."""
            )

        # positive electrode
        self.submodels["positive sei"] = pybamm.sei.NoSEI(self.param, "Positive")

    @property
    def default_parameter_values(self):
        # Default parameter values
        default_params = super().default_parameter_values
        # The mean particle radii for each electrode, taken to be the
        # "Negative particle radius [m]" and "Positive particle radius [m]"
        # provided in the parameter set. These will be the means of the
        # (area-weighted) particle-size distributions f_a_dist_n_dim,
        # f_a_dist_p_dim, provided below.
        R_n_dim = default_params["Negative particle radius [m]"]
        R_p_dim = default_params["Positive particle radius [m]"]

        # Standard deviations (dimensionless)
        sd_a_n = 0.3
        sd_a_p = 0.3

        # Minimum radius in the particle-size distributions (dimensionless).
        R_min_n = 0
        R_min_p = 0

        # Max radius in the particle-size distributions (dimensionless).
        # 5 standard deviations above the mean
        R_max_n = 1 + sd_a_n * 5
        R_max_p = 1 + sd_a_p * 5

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

        # Set the dimensional (area-weighted) particle-size distributions
        def f_a_dist_n_dim(R):
            return lognormal_distribution(R, R_n_dim, sd_a_n * R_n_dim)

        def f_a_dist_p_dim(R):
            return lognormal_distribution(R, R_p_dim, sd_a_p * R_p_dim)

        # Append to default parameters
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


