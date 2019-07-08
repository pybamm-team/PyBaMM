#
# Lead-acid Newman-Tiedemann model
#
import pybamm
from .base_lead_acid_model import BaseModel


class NewmanTiedemann(BaseModel):
    """Porous electrode model for lead-acid, from [1]_.

    References
    ----------
    .. [1] V Sulzer, SJ Chapman, CP Please, DA Howey, and CW Monroe. Faster Lead-Acid
           Battery Simulations from Porous-Electrode Theory: I. Physical Model.
           arXiv preprint arXiv:1902.01771, 2019.


    **Extends:** :class:`pybamm.lead_acid.BaseModel`
    """

    def __init__(self, options=None):
        super().__init__(options)
        self.name = "Newman-Tiedemann model"

        self.set_reactions()
        self.set_current_collector_submodel()
        self.set_interfacial_submodel()
        self.set_porosity_submodel()
        self.set_convection_submodel()
        self.set_electrolyte_submodel()
        self.set_negative_electrode_submodel()
        self.set_positive_electrode_submodel()
        self.set_thermal_submodel()
        self.set_side_reaction_submodels()

        self.build_model()

    def set_current_collector_submodel(self):
        self.submodels["current collector"] = pybamm.current_collector.Uniform(
            self.param, "Negative"
        )

    def set_porosity_submodel(self):
        self.submodels["porosity"] = pybamm.porosity.Full(self.param)

    def set_convection_submodel(self):
        if self.options["convection"] is False:
            self.submodels["convection"] = pybamm.convection.NoConvection(self.param)
        if self.options["convection"] is True:
            self.submodels["convection"] = pybamm.convection.Full(self.param)

    def set_interfacial_submodel(self):
        self.submodels["negative interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Negative"
        )
        self.submodels["positive interface"] = pybamm.interface.lead_acid.ButlerVolmer(
            self.param, "Positive"
        )

    def set_negative_electrode_submodel(self):
        if self.options["surface form"] is False:
            submodel = pybamm.electrode.ohm.Full(self.param, "Negative", self.reactions)
        else:
            submodel = pybamm.electrode.ohm.SurfaceForm(self.param, "Negative")
        self.submodels["negative electrode"] = submodel

    def set_positive_electrode_submodel(self):
        if self.options["surface form"] is False:
            submodel = pybamm.electrode.ohm.Full(self.param, "Positive", self.reactions)
        else:
            submodel = pybamm.electrode.ohm.SurfaceForm(self.param, "Positive")
        self.submodels["positive electrode"] = submodel

    def set_electrolyte_submodel(self):

        electrolyte = pybamm.electrolyte.stefan_maxwell
        surf_form = electrolyte.conductivity.surface_potential_form

        self.submodels["electrolyte diffusion"] = electrolyte.diffusion.Full(
            self.param, self.reactions
        )

        if self.options["surface form"] is False:
            self.submodels["electrolyte conductivity"] = electrolyte.conductivity.Full(
                self.param, self.reactions
            )
        elif self.options["surface form"] == "differential":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullDifferential(self.param, domain, self.reactions)
        elif self.options["surface form"] == "algebraic":
            for domain in ["Negative", "Separator", "Positive"]:
                self.submodels[
                    domain.lower() + " electrolyte conductivity"
                ] = surf_form.FullAlgebraic(self.param, domain, self.reactions)

    def set_side_reaction_submodels(self):
        if "oxygen" in self.options["side reactions"]:
            self.submodels[
                "positive oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.ForwardTafel(self.param, "Positive")
            self.submodels[
                "negative oxygen interface"
            ] = pybamm.interface.lead_acid_oxygen.LeadingOrderDiffusionLimited(
                self.param, "Negative"
            )
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.Full(
                self.param, self.reactions
            )
        else:
            self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(
                self.param
            )

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        # Different solver depending on whether we solve ODEs or DAEs
        if self.options["surface form"] == "differential":
            return pybamm.ScipySolver()
        else:
            return pybamm.ScikitsDaeSolver()
