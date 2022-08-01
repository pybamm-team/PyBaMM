#
# Many-Particle Model (MPM)
#
import pybamm
from .spm import SPM


class MPM(SPM):
    """
    Many-Particle Model (MPM) of a lithium-ion battery with particle-size
    distributions for each electrode, from [1]_.

    Parameters
    ----------
    options : dict, optional
        A dictionary of options to be passed to the model. For a detailed list of
        options see :class:`~pybamm.BatteryModelOptions`.
    name : str, optional
        The name of the model.
    build :  bool, optional
        Whether to build the model on instantiation. Default is True. Setting this
        option to False allows users to change any number of the submodels before
        building the complete model (submodels cannot be changed after the model is
        built).
    Examples
    --------
    >>> import pybamm
    >>> model = pybamm.lithium_ion.MPM()
    >>> model.name
    'Many-Particle Model'

    References
    ----------
    .. [1] TL Kirk, J Evans, CP Please and SJ Chapman. “Modelling electrode
        heterogeneity in lithium-ion batteries: unimodal and bimodal particle-size
        distributions”.
        In: arXiv preprint arXiv:2006.12208 (2020).

    **Extends:** :class:`pybamm.lithium_ion.SPM`
    """

    def __init__(self, options=None, name="Many-Particle Model", build=True):
        # Necessary options
        if options is None:
            options = {"particle size": "distribution", "surface form": "algebraic"}
        elif "particle size" in options and options["particle size"] != "distribution":
            raise pybamm.OptionError(
                "particle size must be 'distribution' for MPM not '{}'".format(
                    options["particle size"]
                )
            )
        elif "surface form" in options and options["surface form"] != "algebraic":
            raise pybamm.OptionError(
                "surface form must be 'algebraic' for MPM not '{}'".format(
                    options["surface form"]
                )
            )
        else:
            options["particle size"] = "distribution"
            options["surface form"] = "algebraic"
        super().__init__(options, name, build)

        pybamm.citations.register("Kirk2020")
        pybamm.citations.register("Kirk2021")

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options["particle"] == "Fickian diffusion":
                submod = pybamm.particle.FickianDiffusion(
                    self.param, domain, self.options, x_average=True
                )
            elif self.options["particle"] == "uniform profile":
                submod = pybamm.particle.XAveragedPolynomialProfile(
                    self.param, domain, self.options
                )
            self.submodels[f"{domain} particle"] = submod

    @property
    def default_parameter_values(self):
        default_params = super().default_parameter_values
        default_params = pybamm.get_size_distribution_parameters(default_params)
        return default_params
