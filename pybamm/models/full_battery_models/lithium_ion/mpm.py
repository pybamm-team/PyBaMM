#
# Many-Particle Model (MPM)
#
import pybamm
from .spm import SPM


class MPM(SPM):
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

    **Extends:** :class:`pybamm.lithium_ion.SPM`
    """

    def __init__(
        self, options=None, name="Many-Particle Model", build=True
    ):
        # Necessary options
        if options is None:
            options = {
                "particle size": "distribution",
                "surface form": "algebraic"
            }
        else:
            options["particle size"] = "distribution"
            options["surface form"] = "algebraic"
        super(SPM, self).__init__(options, name)

        # For degradation models we use the "x-average" form since this is a
        # reduced-order model with uniform current density in the electrodes
        self.x_average = True

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
        pybamm.citations.register("Kirk2021")

    def set_particle_submodel(self):

        if self.options["particle size"] != "distribution":
            raise pybamm.OptionError(
                "particle size must be 'distribution' for MPM not '{}'".format(
                    self.options["particle size"]
                )
            )

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

    @property
    def default_parameter_values(self):
        default_params = super().default_parameter_values
        default_params = pybamm.get_size_distribution_parameters(default_params)
        return default_params


