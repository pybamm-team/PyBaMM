#
# Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_lithium_ion_model import BaseModel


class DFN(BaseModel):
    """
    Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, from [1]_.

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
    >>> model = pybamm.lithium_ion.DFN()
    >>> model.name
    'Doyle-Fuller-Newman model'

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. Journal of The
           Electrochemical Society, 166(15):A3693–A3706, 2019


    **Extends:** :class:`pybamm.lithium_ion.BaseModel`
    """

    def __init__(self, options=None, name="Doyle-Fuller-Newman model", build=True):
        # For degradation models we use the full form since this is a full-order model
        self.x_average = False
        super().__init__(options, name)

        self.set_submodels(build)

        pybamm.citations.register("Doyle1993")

    def set_convection_submodel(self):

        self.submodels[
            "transverse convection"
        ] = pybamm.convection.transverse.NoConvection(self.param, self.options)
        self.submodels[
            "through-cell convection"
        ] = pybamm.convection.through_cell.NoConvection(self.param, self.options)

    def set_intercalation_kinetics_submodel(self):
        for domain in ["Negative", "Positive"]:
            intercalation_kinetics = self.get_intercalation_kinetics(domain)
            self.submodels[domain.lower() + " interface"] = intercalation_kinetics(
                self.param, domain, "lithium-ion main", self.options
            )

    def set_particle_submodel(self):
        for domain in ["negative", "positive"]:
            particle = getattr(self.options, domain)["particle"]
            if particle == "Fickian diffusion":
                self.submodels[f"{domain} particle"] = pybamm.particle.FickianDiffusion(
                    self.param, domain, self.options, x_average=False
                )
            elif particle in [
                "uniform profile",
                "quadratic profile",
                "quartic profile",
            ]:
                self.submodels[
                    f"{domain} particle"
                ] = pybamm.particle.PolynomialProfile(self.param, domain, self.options)

    def set_solid_submodel(self):

        if self.options["surface form"] == "false":
            submod_n = pybamm.electrode.ohm.Full(self.param, "Negative", self.options)
            submod_p = pybamm.electrode.ohm.Full(self.param, "Positive", self.options)
        else:
            submod_n = pybamm.electrode.ohm.SurfaceForm(
                self.param, "Negative", self.options
            )
            submod_p = pybamm.electrode.ohm.SurfaceForm(
                self.param, "Positive", self.options
            )

        self.submodels["negative electrode potential"] = submod_n
        self.submodels["positive electrode potential"] = submod_p

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param, self.options
        )

        if self.options["electrolyte conductivity"] not in ["default", "full"]:
            raise pybamm.OptionError(
                "electrolyte conductivity '{}' not suitable for DFN".format(
                    self.options["electrolyte conductivity"]
                )
            )

        if self.options["surface form"] == "false":
            self.submodels[
                "electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.Full(self.param, self.options)
        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.FullDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.FullAlgebraic

        for domain in ["Negative", "Separator", "Positive"]:
            self.submodels[
                domain.lower() + " surface potential difference"
            ] = surf_model(self.param, domain, self.options)
