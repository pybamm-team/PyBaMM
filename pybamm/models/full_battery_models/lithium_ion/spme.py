#
# Single Particle Model with Electrolyte (SPMe)
#
import pybamm
from .spm import SPM


class SPMe(SPM):
    """
    Single Particle Model with Electrolyte (SPMe) of a lithium-ion battery, from
    [1]_. Inherits most submodels from SPM, only modifies potentials and electrolyte.

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
    >>> model = pybamm.lithium_ion.SPMe()
    >>> model.name
    'Single Particle Model with electrolyte'

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019

    **Extends:** :class:`pybamm.lithium_ion.SPM`
    """

    def __init__(
        self, options=None, name="Single Particle Model with electrolyte", build=True
    ):
        # For degradation models we use the "x-average" form since this is a
        # reduced-order model with uniform current density in the electrodes
        self.x_average = True

        # Initialize with the SPM
        super().__init__(options, name, build)

    def set_convection_submodel(self):

        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param, self.options)
        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param, self.options)

    def set_transport_efficiency_submodels(self):
        self.submodels[
            "electrolyte transport efficiency"
        ] = pybamm.transport_efficiency.Bruggeman(
            self.param, "Electrolyte", self.options, True
        )
        self.submodels[
            "electrode transport efficiency"
        ] = pybamm.transport_efficiency.Bruggeman(
            self.param, "Electrode", self.options, True
        )

    def set_solid_submodel(self):

        self.submodels["negative electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Negative", options=self.options
        )
        self.submodels["positive electrode potential"] = pybamm.electrode.ohm.Composite(
            self.param, "Positive", options=self.options
        )

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if self.options["surface form"] == "false" or self.half_cell:
            if self.options["electrolyte conductivity"] in ["default", "composite"]:
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Composite(
                    self.param, options=self.options
                )
            elif self.options["electrolyte conductivity"] == "integrated":
                self.submodels[
                    "electrolyte conductivity"
                ] = pybamm.electrolyte_conductivity.Integrated(
                    self.param, options=self.options
                )
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.CompositeDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.CompositeAlgebraic

        for domain in ["Negative", "Positive"]:
            self.submodels[
                domain.lower() + " surface potential difference"
            ] = surf_model(self.param, domain)

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param, self.options
        )
