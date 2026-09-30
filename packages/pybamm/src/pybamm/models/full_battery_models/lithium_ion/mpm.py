#
# Many-Particle Model (MPM)
#
import pybamm

from .spm import SPM


class MPM(SPM):
    """
    Many-Particle Model (MPM) of a lithium-ion battery with particle-size
    distributions for each electrode, from :footcite:t:`Kirk2020`.
    See :class:`pybamm.lithium_ion.BaseModel` for more details.

    Examples
    --------
    >>> model = pybamm.lithium_ion.MPM()
    >>> model.name
    'Many-Particle Model'

    """

    def _model_default_options(self, supplied):
        defaults = super()._model_default_options(supplied)
        if "particle size" in supplied and supplied["particle size"] != "distribution":
            raise pybamm.OptionError(
                "particle size must be 'distribution' for MPM not '{}'".format(
                    supplied["particle size"]
                )
            )
        if "surface form" in supplied and supplied["surface form"] == "false":
            raise pybamm.OptionError(
                "surface form must be 'algebraic' or 'differential' for MPM not 'false'"
            )
        defaults.update({"particle size": "distribution", "surface form": "algebraic"})
        return defaults

    def __init__(self, options=None, name="Many-Particle Model", build=True):
        super().__init__(options, name, build)

        pybamm.citations.register("Kirk2020")
        pybamm.citations.register("Kirk2021")

    @property
    def default_parameter_values(self):
        default_params = super().default_parameter_values
        default_params = pybamm.get_size_distribution_parameters(
            default_params, working_electrode=self.options["working electrode"]
        )
        return default_params
