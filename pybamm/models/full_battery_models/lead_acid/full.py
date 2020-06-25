#
# Lead-acid Full model
#
import pybamm
from .base_lead_acid_model import BaseModel


class Full(BaseModel):
    """Porous electrode model for lead-acid, from [1]_, based on the Newman-Tiedemann
    model.

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
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster lead-acid
           battery simulations from porous-electrode theory: Part II. Asymptotic
           analysis. Journal of The Electrochemical Society 166.12 (2019), A2372–A2382.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None, name="Full model", build=True):
        super().__init__(options, name)

        self.set_external_circuit_submodel()
        self.set_interfacial_submodel()
        self.set_porosity_submodel()
        self.set_tortuosity_submodels()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_solid_submodel()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()
        self.set_current_collector_submodel()
        self.set_sei_submodel()

        if build:
            self.build_model()

        pybamm.citations.register("sulzer2019physical")

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.Full(self.param)

    def set_convection_submodel(self):
        if self.options["convection"] is False:
            self.submodels[
                "transverse convection"
            ] = pybamm.convection.transverse.NoConvection(self.param)
            self.submodels[
                "through-cell convection"
            ] = pybamm.convection.through_cell.NoConvection(self.param)
        else:
            if self.options["convection"] == "uniform transverse":
                self.submodels[
                    "transverse convection"
                ] = pybamm.convection.transverse.Uniform(self.param)
            elif self.options["convection"] == "full transverse":
                self.submodels[
                    "transverse convection"
                ] = pybamm.convection.transverse.Full(self.param)
            self.submodels[
                "through-cell convection"
            ] = pybamm.convection.through_cell.Full(self.param)

    def set_interfacial_submodel(self):
        self.submodels["negative interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Negative", "lead-acid main"
        )
        self.submodels["positive interface"] = pybamm.interface.ButlerVolmer(
            self.param, "Positive", "lead-acid main"
        )

    def set_solid_submodel(self):
        if self.options["surface form"] is False:
            submod_n = pybamm.electrode.ohm.Full(self.param, "Negative")
            submod_p = pybamm.electrode.ohm.Full(self.param, "Positive")
        else:
            submod_n = pybamm.electrode.ohm.SurfaceForm(self.param, "Negative")
            submod_p = pybamm.electrode.ohm.SurfaceForm(self.param, "Positive")

        self.submodels["negative electrode"] = submod_n
        self.submodels["positive electrode"] = submod_p

    def set_electrolyte_submodel(self):

        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param
        )

        if self.options["surface form"] is False:
            self.submodels[
                "electrolyte conductivity"
            ] = pybamm.electrolyte_conductivity.Full(self.param)
        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullDifferential(self.param, domain)
        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullAlgebraic(self.param, domain)

    def set_side_reaction_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Full(
                self.param
            )
            self.submodels["positive oxygen interface"] = pybamm.interface.ForwardTafel(
                self.param, "Positive", "lead-acid oxygen"
            )
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.interface.DiffusionLimited(
                self.param, "Negative", "lead-acid oxygen", order="full"
            )
        else:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(
                self.param
            )
            self.submodels["positive oxygen interface"] = pybamm.interface.NoReaction(
                self.param, "Positive", "lead-acid oxygen"
            )
            self.submodels["negative oxygen interface"] = pybamm.interface.NoReaction(
                self.param, "Negative", "lead-acid oxygen"
            )
