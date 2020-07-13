#
# Single Particle Model (SPM)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class PSDModel(BaseModel):
    """Particle-Size Distribution (PSD) Model of a lithium-ion battery, from [1]_.

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
    .. [1] TL Kirk, J Evans, CP Please and SJ Chapman. “Modelling electrode heterogeneity
        in lithium-ion batteries: unimodal and bimodal particle-size distributions”.
        In: arXiv preprint arXiv:2006.12208 (2020).

    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(
        self, options=None, name="Particle-Size Distribution Model", build=True
    ):
        super().__init__(options, name)
        self.options["particle-size distribution"] = True

        # Set length scales for additional domains (particle-size domains)
        self.length_scales.update(
            {
                "negative particle-size domain": self.param.R_n,
                "positive particle-size domain": self.param.R_p,
            }
        )
        # Update standard output variables
        self.set_standard_output_variables()

        # Set submodels
        self.set_external_circuit_submodel()
        self.set_porosity_submodel()
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

        if build:
            self.build_model()

        # pybamm.citations.register("marquis2019asymptotic")

    def set_porosity_submodel(self):

        self.submodels["porosity"] = pybamm.porosity.Constant(self.param)

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
            self.submodels["negative particle"] = pybamm.particle.FickianSinglePSD(
                self.param, "Negative"
            )
            self.submodels["positive particle"] = pybamm.particle.FickianSinglePSD(
                self.param, "Positive"
            )
        elif self.options["particle"] == "fast diffusion":
            self.submodels["negative particle"] = pybamm.particle.FastSinglePSD(
                self.param, "Negative"
            )
            self.submodels["positive particle"] = pybamm.particle.FastSinglePSD(
                self.param, "Positive"
            )

    def set_negative_electrode_submodel(self):

        self.submodels[
            "negative electrode"
        ] = pybamm.electrode.ohm.LeadingOrderSizeDistribution(self.param, "Negative")

    def set_positive_electrode_submodel(self):

        self.submodels[
            "positive electrode"
        ] = pybamm.electrode.ohm.LeadingOrderSizeDistribution(self.param, "Positive")

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["surface form"] is False:
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

    def set_standard_output_variables(self):
        super().set_standard_output_variables()

        # add particle-size variables
        var = pybamm.standard_spatial_vars
        R_n = pybamm.geometric_parameters.R_n
        R_p = pybamm.geometric_parameters.R_p
        self.variables.update(
            {
                "Negative particle size": var.R_variable_n,
                "Negative particle size [m]": var.R_variable_n * R_n,
                "Positive particle size": var.R_variable_p,
                "Positive particle size [m]": var.R_variable_p * R_p,
            }
        )

    ####################
    # Overwrite defaults
    ####################
    @property
    def default_parameter_values(self):
        # Default parameter values
        # Lion parameters left as default parameter set for tests
        default_params = super().default_parameter_values

        # New parameter values
        # Area-weighted standard deviations
        sd_a_n = 0.5
        sd_a_p = 0.3
        sd_a_n_dim = sd_a_n * default_params["Negative particle radius [m]"]
        sd_a_p_dim = sd_a_p * default_params["Positive particle radius [m]"]

        # Max radius in the particle-size distribution (dimensionless)
        R_n_max = max(2, 1 + sd_a_n * 5)
        R_p_max = max(2, 1 + sd_a_p * 5)

        # lognormal area-weighted particle-size distribution

        def lognormal_distribution(R, R_av, sd):
            import numpy as np

            # inputs are particle radius R, the mean R_av, and standard deviation sd
            # inputs can be dimensional or dimensionless
            mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
            sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
            return (
                pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
                / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
                / (R)
            )

        default_params.update(
            {"Negative area-weighted particle-size standard deviation": sd_a_n},
            check_already_exists=False,
        )
        default_params.update(
            {"Negative area-weighted particle-size standard deviation [m]": sd_a_n_dim},
            check_already_exists=False,
        )
        default_params.update(
            {"Positive area-weighted particle-size standard deviation": sd_a_p},
            check_already_exists=False,
        )
        default_params.update(
            {"Positive area-weighted particle-size standard deviation [m]": sd_a_p_dim},
            check_already_exists=False,
        )
        default_params.update(
            {"Negative maximum particle radius": R_n_max}, check_already_exists=False
        )
        default_params.update(
            {"Positive maximum particle radius": R_p_max}, check_already_exists=False
        )
        default_params.update(
            {
                "Negative area-weighted particle-size distribution [m]": lognormal_distribution
            },
            check_already_exists=False,
        )
        default_params.update(
            {
                "Positive area-weighted particle-size distribution [m]": lognormal_distribution
            },
            check_already_exists=False,
        )

        return default_params

    @property
    def default_geometry(self):
        default_geom = super().default_geometry

        # New Spatial Variables
        R_variable_n = pybamm.standard_spatial_vars.R_variable_n
        R_variable_p = pybamm.standard_spatial_vars.R_variable_p

        # append new domains
        default_geom.update(
            {
                "negative particle-size domain": {
                    R_variable_n: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Parameter("Negative maximum particle radius"),
                    }
                },
                "positive particle-size domain": {
                    R_variable_p: {
                        "min": pybamm.Scalar(0),
                        "max": pybamm.Parameter("Positive maximum particle radius"),
                    }
                },
            }
        )
        return default_geom

    @property
    def default_var_pts(self):
        defaults = super().default_var_pts

        # New Spatial Variables
        R_variable_n = pybamm.standard_spatial_vars.R_variable_n
        R_variable_p = pybamm.standard_spatial_vars.R_variable_p
        # add to dictionary
        defaults.update({R_variable_n: 50, R_variable_p: 50})
        return defaults

    @property
    def default_submesh_types(self):
        default_submeshes = super().default_submesh_types

        default_submeshes.update(
            {
                "negative particle-size domain": pybamm.MeshGenerator(
                    pybamm.Uniform1DSubMesh
                ),
                "positive particle-size domain": pybamm.MeshGenerator(
                    pybamm.Uniform1DSubMesh
                ),
            }
        )
        return default_submeshes

    @property
    def default_spatial_methods(self):
        default_spatials = super().default_spatial_methods

        default_spatials.update(
            {
                "negative particle-size domain": pybamm.FiniteVolume(),
                "positive particle-size domain": pybamm.FiniteVolume(),
            }
        )
        return default_spatials
